import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import PCAM
from training.networks import Classifier
from metrics.classifierEval import evaluate_classifier

# Prepare real test split (80/10/10 of 'val')
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
full_ds = PCAM(root='data', split='val', download=True, transform=transform)
N = len(full_ds)
n_train, n_val = int(0.8*N), int(0.1*N)
_,_, test_ds = random_split(full_ds, [n_train, n_val, N-n_train-n_val],
                           generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

def run_eval(ckpt, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = Classifier(num_classes=2).to(device)
    clf.load_state_dict(torch.load(ckpt, map_location=device))
    report, cm = evaluate_classifier(clf, test_loader, device, ['normal','tumor'])
    print(f"=== {name} ===")
    print(report)
    print(cm)

if __name__ == '__main__':
    run_eval('classifier_real.pt',   'REAL ONLY')
    run_eval('classifier_synthetic.pt',  'SYNTHETIC ONLY')
    run_eval('classifier_hybrid.pt', 'HYBRID')