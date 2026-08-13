import os
import uuid
import tempfile

from ai.inference import run_inference
from config import BRATS_DATASET_PATH

# ─────────────────────────────────────────
# AGENT 2 — Scan Analysis Agent
# Job: Take uploaded MRI files and run inference
# ─────────────────────────────────────────
def scan_analysis_agent(
    flair_bytes, t1_bytes, t1ce_bytes, t2_bytes,
    flair_name, t1_name, t1ce_name, t2_name,
    seg_bytes=None, seg_name=None, has_seg=False
):
    print(f"\n[Agent 2] 🔬 Starting scan analysis...")

    # We'll store temp file paths here so we can clean up later
    tmp_files = []

    try:
        # Save each uploaded file to a temp location in /tmp
        def save_tmp(file_bytes, filename):
            # Create unique filename to avoid conflicts
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            tmp_path = os.path.join(tempfile.gettempdir(), unique_name)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
            tmp_files.append(tmp_path)
            print(f"[Agent 2] Saved temp file: {tmp_path}")
            return tmp_path

        # Save all 4 MRI modalities
        flair_path = save_tmp(flair_bytes, flair_name)
        t1_path    = save_tmp(t1_bytes,    t1_name)
        t1ce_path  = save_tmp(t1ce_bytes,  t1ce_name)
        t2_path    = save_tmp(t2_bytes,    t2_name)

        # Build the data dictionary for inference
        data_dict = {
            "flair": flair_path,
            "t1":    t1_path,
            "t1ce": t1ce_path,
            "t2":    t2_path,
        }

        # Save seg file if provided
        if has_seg and seg_bytes is not None:
            seg_path = save_tmp(seg_bytes, seg_name)
            data_dict["seg"] = seg_path
            print(f"[Agent 2] Ground truth segmentation included!")

        # Run the actual model inference
        print(f"[Agent 2] Running model inference...")
        result = run_inference(data_dict, has_seg=has_seg)
        print(f"[Agent 2] ✅ Inference complete!")

        return result

    finally:
        # ALWAYS clean up temp files even if inference fails
        for tmp_path in tmp_files:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"[Agent 2] Cleaned up: {tmp_path}")
            except Exception as e:
                print(f"[Agent 2] Warning: Could not delete {tmp_path}: {e}")

# ─────────────────────────────────────────
# AGENT 2B — Dataset Scan Agent
# Job: Read directly from BraTS dataset folder
# ─────────────────────────────────────────
def dataset_scan_agent(patient_folder):
    print(f"\n[Agent 2B] 📁 Loading from BraTS dataset: {patient_folder}")

    # Build full paths to all files
    base = os.path.join(BRATS_DATASET_PATH, patient_folder)

    # Auto-detect .nii or .nii.gz
    def find_file(base, name):
        for ext in [".nii.gz", ".nii"]:
            path = os.path.join(base, f"{name}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find {name} in {base}")

    flair_path = find_file(base, f"{patient_folder}_flair")
    t1_path    = find_file(base, f"{patient_folder}_t1")
    t1ce_path  = find_file(base, f"{patient_folder}_t1ce")
    t2_path    = find_file(base, f"{patient_folder}_t2")
    seg_path   = find_file(base, f"{patient_folder}_seg")

    # Check all files exist
    for path in [flair_path, t1_path, t1ce_path, t2_path, seg_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Agent 2B] ❌ File not found: {path}")
        print(f"[Agent 2B] ✅ Found: {path}")

    # Build data dict — no temp files needed, read directly!
    data_dict = {
        "flair": flair_path,
        "t1":    t1_path,
        "t1ce": t1ce_path,
        "t2":    t2_path,
        "seg":   seg_path,
    }

    print(f"[Agent 2B] Running model inference...")
    result = run_inference(data_dict, has_seg=True)
    print(f"[Agent 2B] ✅ Dataset scan complete!")

    return result
