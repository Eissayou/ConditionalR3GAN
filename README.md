Install the requirements.txt
If you have a GPU install this too: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Then run this: python gen_images.py --seeds=0-7 --outdir=out --network=network-snapshot-final.pkl
There should be images that are generated in the out folder.