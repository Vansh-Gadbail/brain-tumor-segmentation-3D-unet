import sys
from reports.pdf import generate_pdf_report

patient_info = {
    "patient_id": "P-12345",
    "name": "John Doe",
    "age": 45,
    "gender": "Male"
}

scan_result = {
    "tumor_detected": True,
    "ncr_volume": 12.5,
    "ncr_percent": 10,
    "ed_volume": 45.0,
    "ed_percent": 36,
    "et_volume": 67.5,
    "et_percent": 54,
    "total_volume": 125.0
}

comparison = {
    "previous_date": "2023-01-15",
    "prev_total_volume": 100.0,
    "total_change": {"absolute": "+25.0", "trend": "Increasing"},
    "prev_ncr_volume": 10.0,
    "ncr_change": {"absolute": "+2.5", "trend": "Increasing"},
    "prev_ed_volume": 40.0,
    "ed_change": {"absolute": "+5.0", "trend": "Increasing"},
    "prev_et_volume": 50.0,
    "et_change": {"absolute": "+17.5", "trend": "Increasing"}
}

gemini_report = """1. Findings
There is a noticeable increase in the tumor volume compared to the previous scan. The enhancing tumor region has grown significantly.

2. Recommendations
Immediate clinical correlation and follow-up MRI in 3 months is advised."""

pdf_bytes = generate_pdf_report(
    patient_info=patient_info,
    scan_result=scan_result,
    comparison=comparison,
    gemini_report=gemini_report
)

with open("test_report_output.pdf", "wb") as f:
    f.write(pdf_bytes)

print("Test report generated at test_report_output.pdf")
