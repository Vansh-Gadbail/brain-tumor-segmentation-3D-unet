# database.py
# This file handles all database operations using SQLAlchemy ORM
# We use PostgreSQL - no SQLite anywhere!

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import DATABASE_URL

# Create the base class for all tables
Base = declarative_base()

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a session factory
SessionLocal = sessionmaker(bind=engine)

# ─────────────────────────────────────────
# TABLE 1 — Patients
# ─────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, unique=True, nullable=False)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────
# TABLE 2 — Scans
# ─────────────────────────────────────────
class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"))
    scan_date = Column(String)
    flair_path = Column(String)
    t1_path = Column(String)
    t1ce_path = Column(String)
    t2_path = Column(String)
    tumor_detected = Column(Boolean)
    total_volume = Column(Float)
    ncr_volume = Column(Float)
    ed_volume = Column(Float)
    et_volume = Column(Float)
    ncr_percent = Column(Float)
    ed_percent = Column(Float)
    et_percent = Column(Float)
    overlay_image = Column(Text)
    predicted_image = Column(Text)
    gt_image = Column(Text)
    gemini_report = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────
# FUNCTION — Create tables if they don't exist
# ─────────────────────────────────────────
def init_database():
    print("Initializing database...")
    Base.metadata.create_all(engine)
    print("Database tables created successfully!")

# ─────────────────────────────────────────
# FUNCTION — Check if patient exists
# ─────────────────────────────────────────
def patient_exists(patient_id):
    session = SessionLocal()
    try:
        result = session.query(Patient).filter(Patient.patient_id == patient_id).first()
        return result is not None
    finally:
        session.close()

# ─────────────────────────────────────────
# FUNCTION — Get patient info
# ─────────────────────────────────────────
def get_patient_info(patient_id):
    session = SessionLocal()
    try:
        patient = session.query(Patient).filter(Patient.patient_id == patient_id).first()
        if patient:
            return {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender
            }
        return None
    finally:
        session.close()

# ─────────────────────────────────────────
# FUNCTION — Get latest scan for a patient
# ─────────────────────────────────────────
def get_latest_scan(patient_id):
    session = SessionLocal()
    try:
        scan = session.query(Scan).filter(
            Scan.patient_id == patient_id
        ).order_by(Scan.created_at.desc()).first()

        if scan:
            return {
                "scan_date": scan.scan_date,
                "tumor_detected": scan.tumor_detected,
                "total_volume": scan.total_volume,
                "ncr_volume": scan.ncr_volume,
                "ed_volume": scan.ed_volume,
                "et_volume": scan.et_volume,
                "ncr_percent": scan.ncr_percent,
                "ed_percent": scan.ed_percent,
                "et_percent": scan.et_percent,
            }
        return None
    finally:
        session.close()

# ─────────────────────────────────────────
# FUNCTION — Save a new patient
# ─────────────────────────────────────────
def save_patient(patient_id, name, age, gender):
    session = SessionLocal()
    try:
        if not patient_exists(patient_id):
            patient = Patient(
                patient_id=patient_id,
                name=name,
                age=age,
                gender=gender
            )
            session.add(patient)
            session.commit()
            print(f"Patient {patient_id} saved successfully!")
        else:
            print(f"Patient {patient_id} already exists, skipping save.")
    finally:
        session.close()

# ─────────────────────────────────────────
# FUNCTION — Save a scan result
# ─────────────────────────────────────────
def save_scan(patient_id, scan_data, file_paths):
    session = SessionLocal()
    try:
        scan = Scan(
            patient_id=patient_id,
            scan_date=datetime.utcnow().strftime("%Y-%m-%d"),
            flair_path=file_paths.get("flair", ""),
            t1_path=file_paths.get("t1", ""),
            t1ce_path=file_paths.get("t1ce", ""),
            t2_path=file_paths.get("t2", ""),
            tumor_detected=scan_data.get("tumor_detected", False),
            total_volume=scan_data.get("total_volume", 0.0),
            ncr_volume=scan_data.get("ncr_volume", 0.0),
            ed_volume=scan_data.get("ed_volume", 0.0),
            et_volume=scan_data.get("et_volume", 0.0),
            ncr_percent=scan_data.get("ncr_percent", 0.0),
            ed_percent=scan_data.get("ed_percent", 0.0),
            et_percent=scan_data.get("et_percent", 0.0),
            overlay_image=scan_data.get("overlay_image", ""),
            predicted_image=scan_data.get("predicted_image", ""),
            gt_image=scan_data.get("gt_image", ""),
            gemini_report=scan_data.get("gemini_report", "")
        )
        session.add(scan)
        session.commit()
        print(f"Scan saved for patient {patient_id}!")
    finally:
        session.close()

# ─────────────────────────────────────────
# FUNCTION — Get all scans for a patient
# ─────────────────────────────────────────
def get_all_scans(patient_id):
    session = SessionLocal()
    try:
        scans = session.query(Scan).filter(
            Scan.patient_id == patient_id
        ).order_by(Scan.created_at.asc()).all()

        return [{
            "scan_date": s.scan_date,
            "total_volume": s.total_volume,
            "ncr_volume": s.ncr_volume,
            "ed_volume": s.ed_volume,
            "et_volume": s.et_volume,
        } for s in scans]
    finally:
        session.close()
