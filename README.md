Install the requirements.txt

If you have a GPU install this too: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

I had to gitignore the pickle since it was so big. So download it from here: https://huggingface.co/brownvc/R3GAN-FFHQ-256x256/blob/main/network-snapshot-final.pkl

Then run this: python gen_images.py --seeds=0-7 --outdir=out --network=network-snapshot-final.pkl

There should be images that are generated in the out folder.