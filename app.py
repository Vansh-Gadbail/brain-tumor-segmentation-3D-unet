# app.py
# This is the MAIN UI of the project built with Streamlit
# Think of it as the FACE of the hospital system!

import streamlit as st
import pandas as pd
import base64
import os
from datetime import datetime
import zipfile
import tempfile
import shutil

# ─────────────────────────────────────────
# PAGE CONFIG — Must be first Streamlit command!
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Analysis System",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .tumor-detected {
        color: #d32f2f;
        font-weight: bold;
        font-size: 20px;
    }
    .no-tumor {
        color: #2e7d32;
        font-weight: bold;
        font-size: 20px;
    }
    .metric-card {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #3949ab;
    }
    .grew { color: #d32f2f; font-weight: bold; }
    .shrunk { color: #2e7d32; font-weight: bold; }
    .stable { color: #f57c00; font-weight: bold; }
    .footer {
        text-align: center;
        color: #9e9e9e;
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Analysis </h1>
    <p>AI-Powered MRI Segmentation & Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LAZY IMPORTS — Load heavy modules only when needed
# This prevents black screen on startup!
# ─────────────────────────────────────────
@st.cache_resource
def load_database():
    from database.supabase import init_database
    init_database()
    return True

@st.cache_resource
def load_agents():
    from agents import (
        reception_agent,
        scan_analysis_agent,
        dataset_scan_agent,
        comparison_agent,
        report_generator_agent,
    )
    return (
        reception_agent,
        scan_analysis_agent,
        dataset_scan_agent,
        comparison_agent,
        report_generator_agent,
    )

# Load database in background
with st.spinner("🔄 Initializing system..."):
    try:
        load_database()
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        st.stop()

# Load config
from config import BRATS_DATASET_PATH
from database.supabase import save_scan, get_all_scans
from reports.pdf import generate_pdf_report

# ─────────────────────────────────────────
# SIDEBAR — Patient Information
# ─────────────────────────────────────────
with st.sidebar:
    st.header("👤 Patient Information")

    patient_id = st.text_input(
        "Patient ID *",
        placeholder="e.g. PT001",
        help="Required — unique identifier for the patient"
    )

    patient_name = st.text_input(
        "Patient Name *",
        placeholder="e.g. John Doe",
        help="Required"
    )

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.divider()

    # Color legend
    st.markdown("### 🎨 Color Legend")
    st.markdown("🔴 **NCR** — Necrotic Core")
    st.markdown("🟢 **ED** — Edema")
    st.markdown("🔵 **ET** — Enhancing Tumor")

# ─────────────────────────────────────────
# MAIN AREA — Three Tabs
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload MRI Files", "🗜️ Upload ZIP", "📁 Select from BraTS Dataset"])

# Variables to track what input method is used
uploaded_files_ready = False
zip_ready = False
dataset_patient_ready = False
selected_folder = None
seg_file = None
zip_modalities = {}

# ── TAB 1 — Upload MRI Files ──
with tab1:
    st.subheader("Upload MRI Modalities")

    col1, col2 = st.columns(2)

    with col1:
        flair_file = st.file_uploader(
            "FLAIR Scan *",
            type=["nii", "gz"],
            key="flair"
        )
        t1ce_file = st.file_uploader(
            "T1CE Scan *",
            type=["nii", "gz"],
            key="t1ce"
        )

    with col2:
        t1_file = st.file_uploader(
            "T1 Scan *",
            type=["nii", "gz"],
            key="t1"
        )
        t2_file = st.file_uploader(
            "T2 Scan *",
            type=["nii", "gz"],
            key="t2"
        )

    # Optional ground truth segmentation
    with st.expander("📎 Optional: Upload Ground Truth Segmentation"):
        seg_file = st.file_uploader(
            "Segmentation File (.nii or .nii.gz)",
            type=["nii", "gz"],
            key="seg"
        )
        if seg_file:
            st.success("✅ Ground truth segmentation uploaded!")

    # Show which files are missing
    missing = []
    if not flair_file: missing.append("FLAIR")
    if not t1_file:    missing.append("T1")
    if not t1ce_file:  missing.append("T1CE")
    if not t2_file:    missing.append("T2")

    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)}")
    else:
        st.success("✅ All 4 MRI modalities uploaded!")
        uploaded_files_ready = True

# ── TAB 2 — Upload ZIP ──
with tab2:
    st.subheader("Upload ZIP Archive")
    zip_file = st.file_uploader("Upload ZIP File *", type=["zip"], key="zip_upload")
    
    if zip_file:
        # Extract to a temp directory if not already done for this file
        if "zip_dir" not in st.session_state or st.session_state.get("zip_name") != zip_file.name:
            if "zip_dir" in st.session_state:
                shutil.rmtree(st.session_state["zip_dir"], ignore_errors=True)
            temp_dir = tempfile.mkdtemp()
            st.session_state["zip_dir"] = temp_dir
            st.session_state["zip_name"] = zip_file.name
            
            try:
                zip_file.seek(0)  # Ensure we read from the beginning of the file
                with zipfile.ZipFile(zip_file) as z:
                    z.extractall(temp_dir)
            except zipfile.BadZipFile:
                st.error("❌ Bad ZIP file. Please upload a valid ZIP archive.")
                shutil.rmtree(temp_dir, ignore_errors=True)
                if "zip_dir" in st.session_state:
                    del st.session_state["zip_dir"]
        
        if "zip_dir" in st.session_state:
            temp_dir = st.session_state["zip_dir"]
            # Find all .nii or .nii.gz files recursively
            extracted_files = []
            for root, dirs, files in os.walk(temp_dir):
                # Ignore macOS metadata directories
                if "__MACOSX" in root:
                    continue
                for f in files:
                    # Ignore macOS hidden files
                    if f.startswith("._"):
                        continue
                    
                    f_lower = f.lower()
                    if f_lower.endswith(".nii") or f_lower.endswith(".nii.gz"):
                        extracted_files.append(os.path.join(root, f))
            
            if not extracted_files:
                st.error("❌ No .nii or .nii.gz files found in the ZIP. Unsupported files.")
            else:
                # Auto-detect
                detected = {"flair": [], "t1": [], "t1ce": [], "t2": [], "seg": []}
                for path in extracted_files:
                    name_lower = os.path.basename(path).lower()
                    if "flair" in name_lower: detected["flair"].append(path)
                    elif "t1ce" in name_lower: detected["t1ce"].append(path)
                    elif "t1" in name_lower: detected["t1"].append(path)
                    elif "t2" in name_lower: detected["t2"].append(path)
                    elif "seg" in name_lower: detected["seg"].append(path)
                
                # Check for duplicates or missing
                missing_mods = [k for k in ["flair", "t1", "t1ce", "t2"] if len(detected[k]) == 0]
                duplicates = [k for k, v in detected.items() if len(v) > 1]
                
                if not missing_mods and not duplicates:
                    st.success("✅ All modalities detected successfully.")
                    zip_modalities = {k: v[0] for k, v in detected.items() if v}
                    zip_ready = True
                else:
                    if missing_mods:
                        st.warning(f"⚠️ Missing modalities: {', '.join(missing_mods).upper()}")
                    if duplicates:
                        st.warning(f"⚠️ Duplicate modalities found for: {', '.join(duplicates).upper()}")
                    
                    st.info("Please map the files manually from the extracted contents:")
                    
                    # Show dropdowns
                    file_names = [os.path.relpath(p, temp_dir) for p in extracted_files]
                    options = ["-- Select File --"] + file_names
                    
                    def get_index(mod_key):
                        if len(detected[mod_key]) == 1:
                            return file_names.index(os.path.relpath(detected[mod_key][0], temp_dir)) + 1
                        return 0
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        sel_flair = st.selectbox("FLAIR", options, index=get_index("flair"), key="z_flair")
                        sel_t1ce = st.selectbox("T1CE", options, index=get_index("t1ce"), key="z_t1ce")
                    with col2:
                        sel_t1 = st.selectbox("T1", options, index=get_index("t1"), key="z_t1")
                        sel_t2 = st.selectbox("T2", options, index=get_index("t2"), key="z_t2")
                        
                    sel_seg = st.selectbox("Segmentation (Optional)", options, index=get_index("seg"), key="z_seg")
                    
                    if "-- Select File --" not in [sel_flair, sel_t1, sel_t1ce, sel_t2]:
                        zip_modalities = {
                            "flair": os.path.join(temp_dir, sel_flair),
                            "t1": os.path.join(temp_dir, sel_t1),
                            "t1ce": os.path.join(temp_dir, sel_t1ce),
                            "t2": os.path.join(temp_dir, sel_t2),
                        }
                        if sel_seg != "-- Select File --":
                            zip_modalities["seg"] = os.path.join(temp_dir, sel_seg)
                        zip_ready = True

# ── TAB 3 — BraTS Dataset ──
with tab3:
    st.subheader("Select from BraTS Dataset")

    if not BRATS_DATASET_PATH or not os.path.exists(BRATS_DATASET_PATH):
        st.warning("""
        ⚠️ BraTS dataset path not configured!

        Please set **BRATS_DATASET_PATH** in your `.env` file:
```
        BRATS_DATASET_PATH=C:/Users/Dell/your_brats_folder
```
        Then restart the app.
        """)
    else:
        # List all patient folders
        folders = sorted([
            f for f in os.listdir(BRATS_DATASET_PATH)
            if os.path.isdir(os.path.join(BRATS_DATASET_PATH, f))
        ])

        if not folders:
            st.error("❌ No patient folders found in the dataset path!")
        else:
            selected_folder = st.selectbox(
                "Select Patient Folder",
                folders,
                help="Select a BraTS patient folder to analyze"
            )
            st.info(
                "ℹ️ Ground truth segmentation will be "
                "loaded automatically from the dataset!"
            )
            dataset_patient_ready = True

# ─────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────
st.divider()

# Button is disabled until all required inputs are filled
button_disabled = not (
    patient_id and
    patient_name and
    (uploaded_files_ready or dataset_patient_ready or zip_ready)
)

analyze_clicked = st.button(
    "🔬 Analyze MRI Scans",
    type="primary",
    disabled=button_disabled,
    use_container_width=True
)

if button_disabled:
    st.caption(
        "⚠️ Please fill Patient ID, Patient Name, "
        "and upload all 4 MRI files (or upload ZIP, or select a dataset patient)"
    )

# ─────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────
if analyze_clicked:

    # Load agents only when needed
    (
        reception_agent,
        scan_analysis_agent,
        dataset_scan_agent,
        comparison_agent,
        report_generator_agent,
    ) = load_agents()

    patient_info_result = None
    scan_result = None
    comparison_result = None
    gemini_report = ""

    # ── Agent 1: Reception ──
    with st.status("🏥 Checking patient records...", expanded=True) as status:
        try:
            patient_info_result = reception_agent(
                patient_id, patient_name, age, gender
            )
            

            if patient_info_result["is_returning"] and patient_info_result["previous_scan"]:
                st.write(f"✅ Returning patient found!")
                st.write(
                    f"Last scan: "
                    f"{patient_info_result['previous_scan'].get('scan_date')}"
                )
            else:
                st.write("✅ New patient registered!")
            status.update(label="✅ Patient records checked!", state="complete")
        except Exception as e:
            st.error(f"❌ Error in reception: {e}")
            status.update(label="❌ Reception failed!", state="error")
            st.stop()

    # ── Agent 2 or 2B: Scan Analysis ──
    with st.status(
        "🔬 Running model inference on RTX 3050...",
        expanded=True
    ) as status:
        try:
            if dataset_patient_ready and selected_folder:
                st.write(f"📁 Loading from dataset: {selected_folder}")
                scan_result = dataset_scan_agent(selected_folder)
            elif zip_ready and zip_modalities:
                st.write("🗜️ Processing uploaded ZIP files...")
                def read_bytes(path):
                    with open(path, "rb") as f:
                        return f.read()
                
                scan_result = scan_analysis_agent(
                    flair_bytes=read_bytes(zip_modalities["flair"]),
                    t1_bytes=read_bytes(zip_modalities["t1"]),
                    t1ce_bytes=read_bytes(zip_modalities["t1ce"]),
                    t2_bytes=read_bytes(zip_modalities["t2"]),
                    flair_name=os.path.basename(zip_modalities["flair"]),
                    t1_name=os.path.basename(zip_modalities["t1"]),
                    t1ce_name=os.path.basename(zip_modalities["t1ce"]),
                    t2_name=os.path.basename(zip_modalities["t2"]),
                    seg_bytes=read_bytes(zip_modalities["seg"]) if "seg" in zip_modalities else None,
                    seg_name=os.path.basename(zip_modalities["seg"]) if "seg" in zip_modalities else None,
                    has_seg="seg" in zip_modalities,
                )
            else:
                st.write("📤 Processing uploaded MRI files...")
                scan_result = scan_analysis_agent(
                    flair_bytes=flair_file.read(),
                    t1_bytes=t1_file.read(),
                    t1ce_bytes=t1ce_file.read(),
                    t2_bytes=t2_file.read(),
                    flair_name=flair_file.name,
                    t1_name=t1_file.name,
                    t1ce_name=t1ce_file.name,
                    t2_name=t2_file.name,
                    seg_bytes=seg_file.read() if seg_file else None,
                    seg_name=seg_file.name if seg_file else None,
                    has_seg=seg_file is not None,
                )

            st.write(
                f"✅ Tumor volume detected: "
                f"{scan_result['total_volume']} cm³"
            )
            
            status.update(
                label="✅ Inference complete!",
                state="complete"
            )
        except Exception as e:
            st.error(f"❌ Error in scan analysis: {e}")
            status.update(label="❌ Inference failed!", state="error")
            st.stop()

    # ── Agent 3: Comparison (only for returning patients) ──
    if patient_info_result["is_returning"]:
        with st.status(
            "📊 Comparing with previous scan...",
            expanded=True
        ) as status:
            try:
                comparison_result = comparison_agent(
                    scan_result,
                    patient_info_result["previous_scan"]
                )
                st.write(
                    f"✅ Compared with scan from "
                    f"{comparison_result['previous_date']}"
                )
                status.update(
                    label="✅ Comparison complete!",
                    state="complete"
                )
            except Exception as e:
                st.error(f"❌ Error in comparison: {e}")
                status.update(label="❌ Comparison failed!", state="error")

    # ── Agent 4: Report Generation ──
    with st.status(
        "📝 Generating AI report with Gemini...",
        expanded=True
    ) as status:
        try:
            gemini_report = report_generator_agent(
                patient_info={
                    "patient_id": patient_id,
                    "name": patient_name,
                    "age": age,
                    "gender": gender,
                },
                scan_result=scan_result,
                comparison=comparison_result,
            )
            st.write("✅ AI report generated!")
            status.update(
                label="✅ Report ready!",
                state="complete"
            )
        except Exception as e:
            st.error(f"❌ Error generating report: {e}")
            status.update(label="❌ Report failed!", state="error")

    # ── Save to Database ──
    try:
        db_paths = {}
        if dataset_patient_ready:
            db_paths = {"flair": "", "t1": "", "t1ce": "", "t2": ""}
        elif zip_ready:
            db_paths = {
                "flair": os.path.basename(zip_modalities["flair"]),
                "t1": os.path.basename(zip_modalities["t1"]),
                "t1ce": os.path.basename(zip_modalities["t1ce"]),
                "t2": os.path.basename(zip_modalities["t2"]),
            }
        else:
            db_paths = {
                "flair": flair_file.name if flair_file else "",
                "t1": t1_file.name if t1_file else "",
                "t1ce": t1ce_file.name if t1ce_file else "",
                "t2": t2_file.name if t2_file else "",
            }

        save_scan(
            patient_id=patient_id,
            scan_data={**scan_result, "gemini_report": gemini_report},
            file_paths=db_paths
        )
    except Exception as e:
        st.warning(f"⚠️ Could not save to database: {e}")

    # ── Store in session state ──
    st.session_state["analysis_complete"] = True
    st.session_state["scan_result"] = scan_result
    st.session_state["patient_info_result"] = patient_info_result
    st.session_state["comparison_result"] = comparison_result
    st.session_state["gemini_report"] = gemini_report
    st.session_state["active_patient_id"] = patient_id
    st.session_state["slice_idx"] = scan_result["best_slice"]
    
    # ── Clear temporary files after analysis ──
    if "zip_dir" in st.session_state:
        try:
            shutil.rmtree(st.session_state["zip_dir"], ignore_errors=True)
            del st.session_state["zip_dir"]
            if "zip_name" in st.session_state:
                del st.session_state["zip_name"]
        except Exception as e:
            st.warning(f"Could not clear temporary files: {e}")

# ─────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────
if st.session_state.get("analysis_complete", False) and st.session_state.get("active_patient_id") == patient_id:
    # Load from session state
    scan_result = st.session_state["scan_result"]
    patient_info_result = st.session_state["patient_info_result"]
    comparison_result = st.session_state["comparison_result"]
    gemini_report = st.session_state["gemini_report"]

    st.divider()
    st.header("📊 Analysis Results")

    # Tumor detected banner
    if scan_result["tumor_detected"]:
        st.markdown(
            '<p class="tumor-detected">🔴 TUMOR DETECTED</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="no-tumor">🟢 NO TUMOR DETECTED</p>',
            unsafe_allow_html=True
        )

    # ── 4 Metric Cards ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🔘 Total Volume", f"{scan_result['total_volume']} cm³")
    with col2:
        st.metric("🔴 NCR Volume",
                  f"{scan_result['ncr_volume']} cm³",
                  f"{scan_result['ncr_percent']}%")
    with col3:
        st.metric("🟢 ED Volume",
                  f"{scan_result['ed_volume']} cm³",
                  f"{scan_result['ed_percent']}%")
    with col4:
        st.metric("🔵 ET Volume",
                  f"{scan_result['et_volume']} cm³",
                  f"{scan_result['et_percent']}%")
    with col5:
        st.metric("🧠 AI Confidence",
                f"{scan_result['overall_confidence'] * 100:.2f}%"
            )

    st.divider()

    # ── Interactive MRI Viewer ──
    st.subheader("🖼️ Interactive MRI Slice Viewer")

    if "slice_idx" not in st.session_state:
        st.session_state["slice_idx"] = scan_result["best_slice"]

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("⬅️ Previous Slice", use_container_width=True):
            st.session_state["slice_idx"] = max(0, st.session_state["slice_idx"] - 1)
            st.rerun()
    with col3:
        if st.button("Next Slice ➡️", use_container_width=True):
            st.session_state["slice_idx"] = min(scan_result["max_slice"], st.session_state["slice_idx"] + 1)
            st.rerun()
            
    with col2:
        current_slice = st.slider(
            "Select Slice", 
            0, 
            scan_result["max_slice"], 
            st.session_state["slice_idx"],
            key="slice_slider"
        )
        if current_slice != st.session_state["slice_idx"]:
            st.session_state["slice_idx"] = current_slice
            st.rerun()

    from viewer.visualization import (
        create_flair_image,
        create_predicted_image,
        create_overlay_image,
        create_gt_image,
    )

    idx = st.session_state["slice_idx"]
    raw_flair = scan_result["raw_flair"]
    raw_pred = scan_result["raw_pred"]
    raw_gt = scan_result.get("raw_gt")

    flair_slice = raw_flair[:, :, idx]
    pred_slice = raw_pred[:, :, idx]
    
    dyn_flair = create_flair_image(flair_slice)
    dyn_pred = create_predicted_image(pred_slice)
    dyn_overlay = create_overlay_image(flair_slice, pred_slice)

    image_cols = [("1. Original FLAIR", dyn_flair)]
    if raw_gt is not None:
        gt_slice = raw_gt[:, :, idx]
        image_cols.append(("2. Ground Truth Mask", create_gt_image(gt_slice)))
    image_cols.append(("3. Predicted Mask", dyn_pred))
    image_cols.append(("4. MRI + Overlay", dyn_overlay))

    cols = st.columns(len(image_cols))
    for i, (caption, b64_img) in enumerate(image_cols):
        with cols[i]:
            st.image(
                f"data:image/png;base64,{b64_img}",
                caption=caption,
                use_container_width=True
            )

    # Color legend below images
    st.markdown(
        "**Legend:** 🔴 Red = NCR &nbsp;|&nbsp; "
        "🟢 Green = ED &nbsp;|&nbsp; "
        "🔵 Blue = ET",
        unsafe_allow_html=True
    )

    # ── Comparison Section ──
    if comparison_result:
        st.divider()
        st.subheader(
            f"📈 Comparison with Previous Scan "
            f"({comparison_result['previous_date']})"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tc = comparison_result["total_change"]
            st.metric(
                "Total Volume",
                f"{scan_result['total_volume']} cm³",
                f"{tc['absolute']} cm³ ({tc['trend']})"
            )
        with col2:
            nc = comparison_result["ncr_change"]
            st.metric(
                "NCR Volume",
                f"{scan_result['ncr_volume']} cm³",
                f"{nc['absolute']} cm³ ({nc['trend']})"
            )
        with col3:
            ec = comparison_result["ed_change"]
            st.metric(
                "ED Volume",
                f"{scan_result['ed_volume']} cm³",
                f"{ec['absolute']} cm³ ({ec['trend']})"
            )
        with col4:
            etc = comparison_result["et_change"]
            st.metric(
                "ET Volume",
                f"{scan_result['et_volume']} cm³",
                f"{etc['absolute']} cm³ ({etc['trend']})"
            )

    # ── AI Report ──
    st.divider()
    with st.expander("📝 AI Generated Medical Report", expanded=True):
        st.markdown(gemini_report)

    # ── PDF Download ──
    st.divider()
    try:
        pdf_bytes = generate_pdf_report(
            patient_info={
                "patient_id": patient_id,
                "name": patient_name,
                "age": age,
                "gender": gender,
            },
            scan_result=scan_result,
            comparison=comparison_result,
            gemini_report=gemini_report,
            overlay_b64=scan_result.get("overlay_image", ""),
            predicted_b64=scan_result.get("predicted_image", ""),
            gt_b64=scan_result.get("gt_image", ""),
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"brain_tumor_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Could not generate PDF: {e}")




# ─────────────────────────────────────────
# SCAN HISTORY SECTION
# ─────────────────────────────────────────
if patient_id:
    st.divider()
    st.header("📋 Scan History")

    try:
        all_scans = get_all_scans(patient_id)

        if all_scans:
            df = pd.DataFrame(all_scans)
            df.columns = [
                "Date", "Total (cm³)",
                "NCR (cm³)", "ED (cm³)", "ET (cm³)"
            ]
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Tumor Volume Over Time")
            chart_df = df.set_index("Date")
            st.line_chart(chart_df)
        else:
            st.info("No previous scans found for this patient.")
    except Exception as e:
        st.warning(f"Could not load scan history: {e}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    🧠 Brain Tumor Analysis System &nbsp;|&nbsp;
    AI-Powered Medical Imaging &nbsp;|&nbsp;
    ⚠️ For Research Purposes Only — Not for Clinical Use
</div>
""", unsafe_allow_html=True)
