import h5py
import numpy as np
import matplotlib.pyplot as plt


img_path = r"C:\Users\Lenovo\Downloads\TrainData\img\image_96.h5"
mask_path = r"C:\Users\Lenovo\Downloads\TrainData\mask\mask_9.h5"

# Inspect image
# with h5py.File(img_path, "r") as f:
#     print("IMAGE FILE")
#     print("Keys:", list(f.keys()))

#     for key in f.keys():
#         data = f[key]
#         print(f"  {key}")
#         print(f"    Shape: {data.shape}")
#         print(f"    Dtype: {data.dtype}")

# # Inspect mask
# with h5py.File(mask_path, "r") as f:
#     print("\nMASK FILE")
#     print("Keys:", list(f.keys()))

#     for key in f.keys():
#         data = f[key]
#         print(f"  {key}")
#         print(f"    Shape: {data.shape}")
#         print(f"    Dtype: {data.dtype}")

# with h5py.File(img_path, "r") as f:
#     img = f["img"][:]

# print("Shape:", img.shape)

# for i in range(img.shape[-1]):
#     band = img[:, :, i]

#     print(
#         f"Band {i:2d}: "
#         f"min={band.min():.6f}, "
#         f"max={band.max():.6f}, "
#         f"mean={band.mean():.6f}, "
#         f"std={band.std():.6f}"
#     )
    
# with h5py.File(mask_path, "r") as f:
#     mask = f["mask"][:]

# print("\nMask shape:", mask.shape)
# print("Unique values:", np.unique(mask, return_counts=True))

# import glob
# import os

# mask_files = glob.glob(r"C:\Users\Lenovo\Downloads\TrainData\mask\*.h5")

# total_pixels = 0
# total_landslide = 0
# patches_with_landslide = 0

# for path in mask_files[:100]:
#     with h5py.File(path, "r") as f:
#         mask = f["mask"][:]

#     landslide_pixels = np.sum(mask == 1)

#     total_pixels += mask.size
#     total_landslide += landslide_pixels

#     if landslide_pixels > 0:
#         patches_with_landslide += 1

# print("Patches checked:", min(100, len(mask_files)))
# print("Patches containing landslide:", patches_with_landslide)
# print("Total pixels:", total_pixels)
# print("Landslide pixels:", total_landslide)

# print(
#     "Overall landslide percentage:",
#     (total_landslide / total_pixels) * 100
# )

with h5py.File(img_path, "r") as f:
    img = f["img"][:]

with h5py.File(mask_path, "r") as f:
    mask = f["mask"][:]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Display a few representative channels
channels = [0, 3, 6, 8, 12]

for ax, ch in zip(axes.flat[:5], channels):
    im = ax.imshow(img[:, :, ch], cmap="gray")
    ax.set_title(f"Channel {ch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

# Mask
axes[1, 2].imshow(mask, cmap="gray")
axes[1, 2].set_title("Landslide Mask")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()