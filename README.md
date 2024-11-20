# R3GAN
Code for NeurIPS 2024 paper - The GAN is dead; long live the GAN! A Modern Baseline GAN - by Huang et al.

README coming soon.

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