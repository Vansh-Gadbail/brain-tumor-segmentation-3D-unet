# model_inference.py
# This is the BRAIN of the project!
# Preprocessing EXACTLY matches the training notebook
# Model loads ONCE at startup — not reloaded every time

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import base64
import io
import os

from monai.networks.nets import UNet
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    ToTensord,
    MapTransform,
    Orientationd,
    Spacingd,
    ConcatItemsd,
    SpatialPadd,
)

from config import MODEL_PATH

# ─────────────────────────────────────────
# CUSTOM TRANSFORM — Exactly from your notebook
# Converts BraTS seg labels into 3 binary channels
# ─────────────────────────────────────────
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert BraTS segmentation labels to multi-channel format:
      Channel 0 — TC  (Tumor Core)      = labels 1 + 4
      Channel 1 — WT  (Whole Tumor)     = labels 1 + 2 + 4
      Channel 2 — ET  (Enhancing Tumor) = label 4
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            label = d[key].squeeze(0)
            # TC: label 1 and label 4
            result.append(torch.logical_or(label == 1, label == 4))
            # WT: label 1, 2, and 4
            result.append(
                torch.logical_or(
                    torch.logical_or(label == 1, label == 2),
                    label == 4
                )
            )
            # ET: label 4 only
            result.append(label == 4)
            d[key] = torch.stack(result, dim=0).float()
        return d

# ─────────────────────────────────────────
# DEVICE SETUP — Use GPU if available
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Model] Using device: {device}")
if torch.cuda.is_available():
    print(f"[Model] GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────
# MODEL SETUP — Load ONCE at startup
# ─────────────────────────────────────────
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="BATCH",
    act="RELU",
).to(device)


# Load saved weights if model file exists
if os.path.exists(MODEL_PATH):
    print(f"[Model] Loading weights from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Handle common checkpoint formats.
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

    # First try direct load (this matches your model.pth keys).
    # Fallback to normalized keys for checkpoints saved with wrappers.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model.") and not key.startswith("model."):
                # Keep MONAI UNet's internal "model." prefix when it belongs
                # to the architecture itself.
                pass
            normalized_state_dict[new_key] = value
        model.load_state_dict(normalized_state_dict, strict=True)

    model.eval()
    print("[Model] ✅ Model loaded successfully!")
else:
    print(f"[Model] ⚠️ WARNING: No model found at {MODEL_PATH}")
    print("[Model] Please place model weights at MODEL_PATH")

# ─────────────────────────────────────────
# PREPROCESSING TRANSFORMS
# EXACTLY matches your training notebook!
# ─────────────────────────────────────────
def get_transforms(has_seg=False):
    keys = ["flair", "t1", "t1ce", "t2"]
    all_keys = keys + (["seg"] if has_seg else [])

    transforms_list = [
        # Step 1: Load NIfTI files from disk
        LoadImaged(keys=all_keys),

        # Step 2: Add channel dimension to all
        EnsureChannelFirstd(keys=all_keys),

        # Step 3: Reorient to RAS standard
        Orientationd(keys=all_keys, axcodes="RAS"),

        # Step 4: Resample to 1x1x1mm voxel spacing
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=["bilinear"] * 4 + (["nearest"] if has_seg else [])
        ),

        # Step 5: Normalize MRI intensity (nonzero, per channel)
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),

        # Step 6: Pad to at least 128x128x128
        SpatialPadd(keys=all_keys, spatial_size=(128, 128, 128)),

        # Step 8: Concatenate 4 modalities → shape (4, H, W, D)
        ConcatItemsd(keys=keys, name="image"),

        # Step 9: Convert to PyTorch tensors
        ToTensord(keys=["image"] + (["seg"] if has_seg else [])),
    ]

    # Step 10: Convert seg labels to multi-channel (only if seg provided)
    if has_seg:
        transforms_list.append(
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"])
        )

    return Compose(transforms_list)

# ─────────────────────────────────────────
# HELPER — Convert matplotlib figure → base64 PNG string
# ─────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='black', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

# ─────────────────────────────────────────
# HELPER — Color a segmentation mask slice
# Colors: 0=black, 1=red(NCR), 2=green(ED), 3=blue(ET)
# ─────────────────────────────────────────
def mask_to_colored(mask_slice):
    colored = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
    colored[mask_slice == 1] = [1, 0, 0]   # Red   — NCR
    colored[mask_slice == 2] = [0, 1, 0]   # Green — ED
    colored[mask_slice == 3] = [0, 0, 1]   # Blue  — ET
    return colored

# ─────────────────────────────────────────
# MAIN FUNCTION — Run full inference pipeline
# ─────────────────────────────────────────
def run_inference(data_dict, has_seg=False):
    print("[Inference] Starting inference pipeline...")

    try:
        # Apply preprocessing transforms
        transforms = get_transforms(has_seg=has_seg)
        data = transforms(data_dict)
        print("[Inference] ✅ Preprocessing complete!")

        # Move image to GPU/CPU and add batch dimension
        image = data["image"].unsqueeze(0).to(device)

        # Run sliding window inference
        print("[Inference] Running sliding window inference on GPU...")
        with torch.no_grad():
            output = sliding_window_inference(
                image,
                roi_size=(128, 128, 128),
                sw_batch_size=2,
                predictor=model,
                overlap=0.5
            )

        # Get predicted class per voxel
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        print("[Inference] ✅ Inference complete!")

        # Get ground truth mask if available
        gt_mask = None
        if has_seg and "seg" in data:
            seg = data["seg"].cpu().numpy()
            # Reconstruct label map from multi-channel:
            # Channel 0=TC, 1=WT, 2=ET
            gt_mask = np.zeros(seg.shape[1:], dtype=np.uint8)
            gt_mask[seg[1] > 0.5] = 2   # ED (WT but not TC)
            gt_mask[seg[0] > 0.5] = 1   # NCR (TC)
            gt_mask[seg[2] > 0.5] = 3   # ET

        # Get FLAIR slice for visualization (first channel)
        flair_img = data["image"][0].cpu().numpy()

        # ─────────────────────────────────────────
        # Find best axial slice = most tumor voxels
        # ─────────────────────────────────────────
        tumor_per_slice = [
            (pred_mask[:, :, i] > 0).sum()
            for i in range(pred_mask.shape[2])
        ]
        best_slice = int(np.argmax(tumor_per_slice))
        print(f"[Inference] Best visualization slice: {best_slice}")

        # ─────────────────────────────────────────
        # Calculate volumes
        # Each voxel = 1mm³ → divide by 1000 for cm³
        # ─────────────────────────────────────────
        ncr_voxels   = int((pred_mask == 1).sum())
        ed_voxels    = int((pred_mask == 2).sum())
        et_voxels    = int((pred_mask == 3).sum())
        total_voxels = ncr_voxels + ed_voxels + et_voxels

        ncr_volume   = round(ncr_voxels / 1000, 2)
        ed_volume    = round(ed_voxels / 1000, 2)
        et_volume    = round(et_voxels / 1000, 2)
        total_volume = round(total_voxels / 1000, 2)

        # Percentages
        ncr_percent = round(ncr_voxels / total_voxels * 100, 1) if total_voxels > 0 else 0.0
        ed_percent  = round(ed_voxels  / total_voxels * 100, 1) if total_voxels > 0 else 0.0
        et_percent  = round(et_voxels  / total_voxels * 100, 1) if total_voxels > 0 else 0.0

        tumor_detected = total_voxels > 0
        print(f"[Inference] Total tumor volume: {total_volume} cm³")

        # ─────────────────────────────────────────
        # Generate visualization images as base64
        # ─────────────────────────────────────────
        flair_slice = flair_img[:, :, best_slice]
        pred_slice  = pred_mask[:, :, best_slice]

        # Image 1 — Original FLAIR (grayscale)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('black')
        ax.imshow(flair_slice.T, cmap='gray', origin='lower')
        ax.axis('off')
        flair_b64 = fig_to_base64(fig)

        # Image 2 — Predicted segmentation mask (colored)
        pred_colored = mask_to_colored(pred_slice)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('black')
        ax.imshow(pred_colored.transpose(1, 0, 2), origin='lower')
        ax.axis('off')
        predicted_b64 = fig_to_base64(fig)

        # Image 3 — FLAIR + predicted overlay
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('black')
        ax.imshow(flair_slice.T, cmap='gray', origin='lower')
        ax.imshow(pred_colored.transpose(1, 0, 2), alpha=0.6, origin='lower')
        ax.axis('off')
        overlay_b64 = fig_to_base64(fig)

        # Image 4 — Ground truth mask (only if seg provided)
        gt_b64 = ""
        if gt_mask is not None:
            gt_slice   = gt_mask[:, :, best_slice]
            gt_colored = mask_to_colored(gt_slice)
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('black')
            ax.imshow(gt_colored.transpose(1, 0, 2), origin='lower')
            ax.axis('off')
            gt_b64 = fig_to_base64(fig)

        print("[Inference] ✅ All images generated!")

        # Return everything
        return {
            "tumor_detected":  tumor_detected,
            "total_volume":    total_volume,
            "ncr_volume":      ncr_volume,
            "ed_volume":       ed_volume,
            "et_volume":       et_volume,
            "ncr_percent":     ncr_percent,
            "ed_percent":      ed_percent,
            "et_percent":      et_percent,
            "flair_image":     flair_b64,
            "predicted_image": predicted_b64,
            "overlay_image":   overlay_b64,
            "gt_image":        gt_b64,
        }

    except Exception as e:
        print(f"[Inference] ❌ ERROR: {e}")
        raise e