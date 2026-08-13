import os
import uuid
import tempfile
from datetime import datetime

from database.supabase import (
    patient_exists,
    get_latest_scan,
    get_patient_info,
    save_patient,
    save_scan,
)

# ─────────────────────────────────────────
# AGENT 1 — Reception Agent
# Job: Check if patient is new or returning
# ─────────────────────────────────────────
def reception_agent(patient_id, name, age, gender):
    print(f"\n[Agent 1] 🏥 Checking records for patient: {patient_id}")

    # Check if patient exists in database
    is_returning = patient_exists(patient_id)

    previous_scan = None
    if is_returning:
        print(f"[Agent 1] ✅ Returning patient found!")
        previous_scan = get_latest_scan(patient_id)
        if previous_scan:
            print(f"[Agent 1] Last scan date: {previous_scan.get('scan_date')}")
        else:
            is_returning = False
            print(f"[Agent 1] No previous scans found, treating as new patient!")
    else:
        print(f"[Agent 1] 🆕 New patient — saving to database...")
        save_patient(patient_id, name, age, gender)
        print(f"[Agent 1] ✅ Patient saved!")

    return {
        "patient_id":    patient_id,
        "name":          name,
        "age":           age,
        "gender":        gender,
        "is_returning":  is_returning,
        "previous_scan": previous_scan,
    }
