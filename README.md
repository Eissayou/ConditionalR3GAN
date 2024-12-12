## The GAN is dead; long live the GAN! A Modern Baseline GAN (R3GAN)<br><sub>Official PyTorch implementation of the NeurIPS 2024 paper</sub>

![Teaser image](./doc/teaser.png)

**The GAN is dead; long live the GAN! A Modern Baseline GAN**<br>
Nick Huang, Aaron Gokaslan, Volodymyr Kuleshov, James Tompkin
<br>https://openreview.net/forum?id=OrtN9hPP7V<br>

Abstract: *There is a widely-spread claim that GANs are difficult to train, and GAN architectures in the literature are littered with empirical tricks. We provide evidence against this claim and build a modern GAN baseline in a more principled manner. First, we derive a well-behaved regularized relativistic GAN loss that addresses issues of mode dropping and non-convergence that were previously tackled via a bag of ad-hoc tricks. We analyze our loss mathematically and prove that it admits local convergence guarantees, unlike most existing relativistic losses. Second, this loss allows us to discard all ad-hoc tricks and replace outdated backbones used in common GANs with modern architectures. Using StyleGAN2 as an example, we present a roadmap of simplification and modernization that results in a new minimalist baseline. Despite being simple, our approach surpasses StyleGAN2 on FFHQ, ImageNet, CIFAR, and Stacked MNIST datasets, and compares favorably against state-of-the-art GANs and diffusion models.*







To reproduce paper results:
```
# CIFAR10
python train.py --outdir=./training-runs --data=./datasets/cifar10.zip --gpus=8 --batch=512 --mirror=1 --aug=1 --cond=1 --preset=CIFAR10 --tick=1 --snap=200

# FFHQ 64x64
python train.py --outdir=./training-runs --data=./datasets/ffhq-64x64.zip --gpus=8 --batch=256 --mirror=1 --aug=1 --preset=FFHQ-64 --tick=1 --snap=200

# FFHQ 256x256
python train.py --outdir=./training-runs --data=./datasets/ffhq-256x256.zip --gpus=8 --batch=256 --mirror=1 --aug=1 --preset=FFHQ-256 --tick=1 --snap=200

# ImageNet 32x32
python train.py --outdir=./training-runs --data=./datasets/imagenet-32x32.zip --gpus=32 --batch=4096 --mirror=1 --aug=1 --cond=1 --preset=ImageNet-32 --tick=1 --snap=200

# Imagenet 64x64
python train.py --outdir=./training-runs --data=./datasets/imagenet-64x64.zip --gpus=64 --batch=4096 --mirror=1 --aug=1 --cond=1 --preset=ImageNet-64 --tick=1 --snap=200
```