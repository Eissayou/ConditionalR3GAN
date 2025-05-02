### Requirements

```bash
pip install -r requirements.txt
# GPU users (CUDA 11.8 wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔍 Quick Demo – Generate a Few Images

1. **Grab the pretrained weights** (omitted from the repo because of size):

   [https://huggingface.co/brownvc/R3GAN-FFHQ-256x256/blob/main/network-snapshot-final.pkl](https://huggingface.co/brownvc/R3GAN-FFHQ-256x256/blob/main/network-snapshot-final.pkl)

2. **Generate seeds 0‑7** into `out/`:

   ```bash
   python gen_images.py --seeds 0-7 \
                        --outdir out \
                        --network network-snapshot-final.pkl
   ```

   The resulting PNGs land in `out/`.

---

## 🏋️‍♂️  Train on the PCam Dataset (conditional)

```bash
python train.py \
  --outdir training-runs \          # snapshots/logs
  --data   pcam_dataset.zip \       # StyleGAN‑ready zip (see next section)
  --gpus   1 \                      # #GPUs
  --batch  32 \                     # total batch‑size
  --preset CIFAR10 \                # config widths (works for 96² too)
  --cond   True                     # enable class‑conditioning
```

---

## 🛠  Preparing **your own** PCam dataset

> **TL;DR** download the `.h5` patches → convert to `pcam_images/` PNGs → build the StyleGAN zip.


````markdown
1. Download the two validation files and **ungzip** them:

```bash
# Linux / macOS / WSL
gunzip camelyonpatch_level_2_split_valid_x.h5.gz
gunzip camelyonpatch_level_2_split_valid_y.h5.gz
# Windows → use 7‑Zip or Git Bash with the same commands
````

After extraction you should have:

| resulting file                           | purpose                      |
| ---------------------------------------- | ---------------------------- |
| `camelyonpatch_level_2_split_valid_x.h5` | image patches (validation X) |
| `camelyonpatch_level_2_split_valid_y.h5` | labels (validation Y)        |

```

These plain `.h5` files are what `prepare_pcam_dataset.sh` expects.
```

2. **Extract PNGs + labels**

```bash
bash prepare_pcam_dataset.sh       # writes PNGs into pcam_images/
```

3. **Create the StyleGAN zip**

```bash
python dataset_tool.py \
  --source      pcam_images \      # folder of 96×96 PNGs
  --dest        pcam_dataset.zip \ # output zip
  --resolution  96x96 \
  --transform   center-crop        # (omit if already 96×96)
```

4. **Train (conditional example)**

```bash
python train.py \
  --outdir training-runs \
  --data   pcam_dataset.zip \
  --gpus   1 \
  --batch  32 \
  --preset CIFAR10 \
  --cond   True
```

---

> **Windows users:**
> replace the `\` line‑continuations with `^` (CMD) or back‑tick `` ` `` (PowerShell), or just place each command on one long line.
