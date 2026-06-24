"""Downscale GT volume NPZ files from 1024^3 to 512^3 (trilinear, in place)."""
import glob
import os

import numpy as np
from scipy.ndimage import zoom

files = sorted(glob.glob("data/kelvin/lattice_*.npz"))
print(f"Found {len(files)} files")

for f in files:
    d = np.load(f)
    vol = d["vol"]
    if vol.shape == (512, 512, 512):
        print(f"{os.path.basename(f)}: already 512, skipping")
        continue
    assert vol.shape == (1024, 1024, 1024), f"{f}: unexpected shape {vol.shape}"
    out = zoom(vol, 0.5, order=1)
    out = out.astype(np.uint8)
    np.savez_compressed(f, vol=out)
    print(f"{os.path.basename(f)}: {vol.shape} -> {out.shape}")

print("Done")
