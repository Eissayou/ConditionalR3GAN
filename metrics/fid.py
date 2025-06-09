import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid(real_loader, generator, device, latent_dim, num_classes):
    fid = FrechetInceptionDistance().to(device)
    generator.eval()
    with torch.no_grad():
        for real_imgs, _ in real_loader:
            real = real_imgs.to(device)
            real_uint8 = ((real*0.5+0.5)*255).clamp(0,255).to(torch.uint8)
            fid.update(real_uint8, real=True)
            b = real.size(0)
            z = torch.randn(b, latent_dim, device=device)
            y_int = torch.randint(0, num_classes, (b,), device=device)
            y = torch.zeros(b, num_classes, device=device)
            y.scatter_(1, y_int.unsqueeze(1), 1)
            fake = generator(z, y)
            fake_uint8 = ((fake*0.5+0.5)*255).clamp(0,255).to(torch.uint8)
            fid.update(fake_uint8, real=False)
    return fid.compute().item()