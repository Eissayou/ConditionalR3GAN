#!/bin/bash

# ========================
# CONFIGURATION
# ========================
TRAIN_X_PATH="camelyonpatch_level_2_split_valid_x.h5"
TRAIN_Y_PATH="camelyonpatch_level_2_split_valid_y.h5"
OUT_DIR="pcam_images"
DATASET_ZIP="pcam_dataset.zip"

# ========================
# STEP 0: Check for .gz files and unzip if needed
# ========================
echo "Checking for compressed (.gz) files..."

for file in "$TRAIN_X_PATH" "$TRAIN_Y_PATH"; do
    if [[ ! -f "$file" && -f "$file.gz" ]]; then
        echo "Found compressed file: $file.gz, unzipping..."
        gunzip "$file.gz"
    fi
done

# ========================
# STEP 1: Create output directory
# ========================
echo "Creating output folder: $OUT_DIR"
mkdir -p "$OUT_DIR"

# ========================
# STEP 2: Python script to extract images and labels
# ========================
echo "Extracting images and labels..."

/opt/anaconda3/envs/r3gan/bin/python - <<END
import h5py
import os
import json
from PIL import Image
import numpy as np

x_path = "$TRAIN_X_PATH"
y_path = "$TRAIN_Y_PATH"
out_dir = "$OUT_DIR"

x_h5 = h5py.File(x_path, 'r')
y_h5 = h5py.File(y_path, 'r')

x = x_h5['x']
y = y_h5['y']

dataset_json = {'labels': []}

for i in range(len(x)):
    img = Image.fromarray(x[i])

    subfolder = os.path.join(out_dir, f"{i//10000:05d}")
    os.makedirs(subfolder, exist_ok=True)

    filename = f"img{i:08d}.png"
    filepath = os.path.join(subfolder, filename)
    img.save(filepath)

    dataset_json['labels'].append([f"{i//10000:05d}/{filename}", int(y[i][0])])

with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f)
END

echo "Extraction complete."

# ========================
# STEP 3: Convert to zip format
# ========================
echo "Running dataset_tool.py to create zip..."

python3 dataset_tool.py --source="$OUT_DIR" --dest="$DATASET_ZIP" --resolution=96x96

echo "Dataset ready at: $DATASET_ZIP"

# ========================
# DONE
# ========================
echo "All done! You can now train with: "
echo "python3 train.py --outdir=training-runs --data=$DATASET_ZIP --gpus=1 --batch=32 --preset=CIFAR10 --cond=True"
