from datetime import date

import httpx
from fastmcp.tools import tool

_HAPI_BASE = "https://hapi.fhir.org/baseR4"

# Realistic synthetic patient — 76yo woman with new AF, CKD stage 3b, T2DM, on amiodarone.
# This profile is intentionally clinically rich: eGFR constrains anticoagulant dosing,
# amiodarone creates a drug-drug interaction with DOACs, and penicillin allergy rules out
# certain prophylaxis options. Good for demoing patient-aware evidence retrieval.
_DEMO_PATIENT: dict = {
    "patient_id": "demo-76F-AFib-CKD",
    "name": "Jane Doe (Synthetic)",
    "age": 76,
    "sex": "female",
    "weight_kg": 52,
    "height_cm": 158,
    "conditions": [
        {"code": "427003000", "display": "Atrial fibrillation (new onset)", "onset": "2026-04"},
        {"code": "709044004", "display": "Chronic kidney disease stage 3b", "onset": "2022-06"},
        {"code": "44054006",  "display": "Type 2 diabetes mellitus", "onset": "2018-03"},
        {"code": "73211009",  "display": "Essential hypertension", "onset": "2015-01"},
    ],
    "medications": [
        {"name": "Metformin",     "dose": "1000 mg", "frequency": "twice daily"},
        {"name": "Amiodarone",    "dose": "200 mg",  "frequency": "once daily"},
        {"name": "Lisinopril",    "dose": "10 mg",   "frequency": "once daily"},
        {"name": "Atorvastatin",  "dose": "40 mg",   "frequency": "once daily at bedtime"},
    ],
    "labs": [
        {"test": "eGFR",              "value": "38",  "unit": "mL/min/1.73m²", "date": "2026-04-15", "flag": "LOW"},
        {"test": "Serum creatinine",  "value": "1.8", "unit": "mg/dL",          "date": "2026-04-15"},
        {"test": "HbA1c",             "value": "7.8", "unit": "%",              "date": "2026-04-15", "flag": "HIGH"},
        {"test": "Potassium",         "value": "4.1", "unit": "mEq/L",          "date": "2026-04-15"},
        {"test": "INR",               "value": "1.1", "unit": "",               "date": "2026-04-01"},
        {"test": "TSH",               "value": "2.3", "unit": "mIU/L",          "date": "2026-03-01"},
    ],
    "allergies": ["Penicillin (anaphylaxis)", "Sulfa drugs (rash)"],
    "note": "SYNTHETIC DEMO PATIENT — no real PHI",
}


@tool()
async def get_patient_context(patient_id: str | None = None) -> dict:
    """Retrieve the current patient's clinical context for evidence-based reasoning.

    Returns demographics, active conditions, current medications, recent laboratory
    values, and known allergies. When called without a patient_id, returns a synthetic
    demo patient representative of a complex real-world case.

    In a SHARP-enabled deployment, the Prompt Opinion agent host forwards the FHIR
    patient token on every call; this tool reads that context and surfaces it in a
    structured form suitable for passing to answer_clinical_question.
    """
    if patient_id is None:
        return _DEMO_PATIENT

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{_HAPI_BASE}/Patient/{patient_id}",
                headers={"Accept": "application/fhir+json"},
            )
            if resp.status_code == 404:
                return {
                    **_DEMO_PATIENT,
                    "patient_id": patient_id,
                    "note": f"Patient {patient_id} not found on FHIR server — returning demo data",
                }
            resp.raise_for_status()
            resource = resp.json()
    except Exception:
        return {
            **_DEMO_PATIENT,
            "patient_id": patient_id,
            "note": "FHIR server unreachable — returning demo data",
        }

    name_parts = resource.get("name", [{}])[0]
    given = " ".join(name_parts.get("given", []))
    family = name_parts.get("family", "")

    age: int | None = None
    birth_date = resource.get("birthDate", "")
    if birth_date:
        try:
            birth = date.fromisoformat(birth_date)
            age = (date.today() - birth).days // 365
        except ValueError:
            pass

    return {
        "patient_id": patient_id,
        "name": f"{given} {family}".strip() or "Unknown",
        "age": age,
        "sex": resource.get("gender"),
        "weight_kg": None,
        "height_cm": None,
        "conditions": [],
        "medications": [],
        "labs": [],
        "allergies": [],
        "note": "Basic demographics from FHIR Patient resource. Conditions/meds/labs require additional resource queries.",
    }
