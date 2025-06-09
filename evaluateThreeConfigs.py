import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import PCAM

import dnnlib, legacy
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from training.networks import Classifier
from synthetic import SyntheticPCamDataset


def compute_fid_datasets(device, loader_real, loader_fake):
    fid = FrechetInceptionDistance().to(device)
    with torch.no_grad():
        # real images
        for imgs, _ in loader_real:
            imgs = imgs.to(device)
            uint8 = ((imgs * 0.5 + 0.5) * 255.0).clamp(0,255).to(torch.uint8)
            fid.update(uint8, real=True)
        # fake images
        for imgs, _ in loader_fake:
            imgs = imgs.to(device)
            uint8 = ((imgs * 0.5 + 0.5) * 255.0).clamp(0,255).to(torch.uint8)
            fid.update(uint8, real=False)
    return fid.compute().item()


def extract_classifier_features(clf, loader, device):
    feats = []
    clf.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            f = clf.extract_features(imgs).cpu()
            feats.append(f)
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3gan_pkl",      required=True,
                        help="Path to network-snapshot-*.pkl for R3GAN")
    parser.add_argument("--clf_checkpoint", required=True,
                        help="Path to classifier_resnet18_final.pt")
    parser.add_argument("--data_dir",       default="data",
                        help="Root for PCAM data (torchvision.datasets.PCAM)")
    parser.add_argument("--out_dir",        default="out",
                        help="Directory to save FID results and t-SNE plots")
    parser.add_argument("--z_dim",    type=int, default=64,
                        help="Latent dimensionality used in R3GAN")
    parser.add_argument("--n_classes",type=int, default=2,
                        help="Number of conditioning classes in R3GAN")
    parser.add_argument("--batch_size",type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare transforms and PCAM splits
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = PCAM(root=args.data_dir, split="val", download=True, transform=transform)
    N = len(full_ds)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    real_train, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load R3GAN generator
    with dnnlib.util.open_url(args.r3gan_pkl) as f:
        nets = legacy.load_network_pkl(f)
    G = nets['G_ema'].to(device).eval()

    # Prepare synthetic train set
    synth_ds = SyntheticPCamDataset(
        G, latent_dim=args.z_dim, num_classes=args.n_classes,
        length=len(real_train), transform=transform, device=device
    )

    # Load classifier
    clf = Classifier(num_classes=args.n_classes).to(device)
    clf.load_state_dict(torch.load(args.clf_checkpoint, map_location=device))

    # Define three configurations
    configs = [
        ('real',      real_train),
        ('synthetic', synth_ds),
        ('hybrid',    ConcatDataset([real_train, synth_ds])),
    ]

    for name, train_ds in configs:
        print(f"\n=== Configuration: {name.upper()} ===")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # Compute FID between train and test sets
        fid = compute_fid_datasets(device, train_loader, test_loader)
        print(f"FID ({name} vs. test): {fid:.4f}")

        # Extract classifier features for train and test
        feats_train = extract_classifier_features(clf, train_loader, device)
        feats_test  = extract_classifier_features(clf, test_loader,  device)

        # t-SNE
        X = np.vstack([feats_train.numpy(), feats_test.numpy()])
        y = np.array([0]*len(feats_train) + [1]*len(feats_test))
        X_emb = TSNE(n_components=2).fit_transform(X)

        plt.figure(figsize=(6,6))
        plt.scatter(X_emb[y==0,0], X_emb[y==0,1], label='train', alpha=0.6)
        plt.scatter(X_emb[y==1,0], X_emb[y==1,1], label='test',  alpha=0.6)
        plt.legend()
        plt.title(f"t-SNE: {name} vs. test")
        out_path = os.path.join(args.out_dir, f"{name}_tsne.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved t-SNE plot: {out_path}")

if __name__ == '__main__':
    main()
