import asyncio
import json
import re

import httpx
from google import genai
from google.genai import types
from fastmcp.tools import tool

_CLIENT = genai.Client()
_MODEL = "gemini-3.1-flash-lite-preview"
_OPENFDA = "https://api.fda.gov/drug/label.json"

_ACTION_ICON = {"OK": "✓", "REDUCE": "⚠️", "AVOID": "🚫", "MONITOR": "👁️"}

_RENAL_SYSTEM = """You are a clinical pharmacist reviewing FDA prescribing information for renal dose adjustments.

Given a patient's renal function (eGFR in mL/min/1.73m² and estimated CrCl in mL/min) and FDA label sections for each medication, return the dose adjustment requirement for every drug at this specific renal function level.

Return ONLY valid JSON:
{
  "medications": [
    {
      "drug": "drug name as provided",
      "action": "OK|REDUCE|AVOID|MONITOR",
      "recommendation": "specific actionable sentence for this patient's eGFR/CrCl",
      "threshold": "the eGFR or CrCl threshold from the label that triggers this action",
      "label_found": true
    }
  ]
}

action:
- OK: no dose adjustment required at this renal function level
- REDUCE: explicit dose reduction recommended by FDA label at this level
- AVOID: contraindicated or strongly not recommended at this level
- MONITOR: use with caution; increased monitoring of renal function or drug levels advised

If no label data was found for a drug, set label_found=false and action=MONITOR with a generic recommendation.
Always cite the specific numeric threshold from the label (e.g. "eGFR < 30 mL/min"), not just "use with caution"."""


def _cockcroft_gault(age: int, weight_kg: float, creatinine_mg_dl: float, sex: str) -> float:
    crcl = ((140 - age) * weight_kg) / (72 * creatinine_mg_dl)
    if sex and sex.lower() in ("f", "female"):
        crcl *= 0.85
    return round(crcl, 1)


def _extract_lab_value(labs: list, *keywords: str) -> float | None:
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        test_name = lab.get("test", "").lower()
        if any(kw.lower() in test_name for kw in keywords):
            try:
                return float(str(lab.get("value", "")).replace(",", ""))
            except (ValueError, TypeError):
                continue
    return None


def _ckd_stage(egfr: float) -> str:
    if egfr >= 90:
        return "G1 (normal or high)"
    if egfr >= 60:
        return "G2 (mildly decreased)"
    if egfr >= 45:
        return "G3a (mildly-moderately decreased)"
    if egfr >= 30:
        return "G3b (moderately-severely decreased)"
    if egfr >= 15:
        return "G4 (severely decreased)"
    return "G5 (kidney failure)"


async def _fetch_dosing_label(drug: str, client: httpx.AsyncClient) -> str | None:
    for search_field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            resp = await client.get(
                _OPENFDA,
                params={"search": f'{search_field}:"{drug}"', "limit": 1},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    r = results[0]
                    # Combine dosage + warnings — renal info can be in either section
                    dosing = " ".join(r.get("dosage_and_administration", []))
                    warnings = " ".join(r.get("warnings_and_precautions", []))
                    combined = f"{dosing} {warnings}".strip()
                    return combined[:3000] if combined else None
        except Exception:
            continue
    return None


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"medications": []}


def _to_markdown(
    egfr: float | None,
    crcl: float | None,
    ckd_stage: str,
    crcl_formula: str,
    medications: list[dict],
) -> str:
    lines: list[str] = []

    lines += ["## Renal Dose Review", ""]

    if egfr is not None:
        lines.append(f"**eGFR:** {egfr} mL/min/1.73m² — CKD Stage {ckd_stage}")
    if crcl is not None:
        lines.append(f"**Estimated CrCl (Cockcroft-Gault):** {crcl} mL/min  ")
        lines.append(f"*Formula: {crcl_formula}*")
    lines.append("")

    if not medications:
        lines += ["No medications to review.", ""]
    else:
        lines += ["| Medication | Action | Recommendation |", "|---|---|---|"]
        for m in medications:
            icon = _ACTION_ICON.get(m.get("action", ""), "")
            action = m.get("action", "")
            drug = m.get("drug", "").title()
            rec = m.get("recommendation", "")
            label_note = "" if m.get("label_found", True) else " *(label not found)*"
            lines.append(f"| {drug} | {icon} {action} | {rec}{label_note} |")
        lines.append("")

        avoid_reduce = [m for m in medications if m.get("action") in ("AVOID", "REDUCE")]
        if avoid_reduce:
            lines += ["### Medications Requiring Action", ""]
            for m in avoid_reduce:
                icon = _ACTION_ICON[m["action"]]
                lines += [
                    f"**{icon} {m['drug'].title()} — {m['action']}**  ",
                    f"Threshold: {m.get('threshold', 'see label')}  ",
                    f"Recommendation: {m.get('recommendation', '')}",
                    "",
                ]

    lines += [
        "---",
        "⚠️ *Decision support only. Verify against current prescribing information "
        "and reassess if renal function changes.*",
    ]
    return "\n".join(lines)


@tool()
async def check_renal_dosing(
    patient_id: str | None = None,
    override_medications: list[str] | None = None,
) -> dict:
    """Review all patient medications for renal dose adjustments based on current eGFR.

    Calculates eGFR using the CKD-EPI value from labs (or Cockcroft-Gault from creatinine,
    age, weight, and sex if not available). Then fetches the FDA prescribing label for every
    medication and identifies which require dose reduction, avoidance, or monitoring at the
    patient's current renal function level.

    Returns both `markdown` (summary table for chat) and `structured` (for downstream agents).

    Args:
        patient_id: FHIR patient ID — pulls medications and labs automatically.
        override_medications: Explicit med list; used instead of FHIR if provided.
    """
    from tools.fhir_patient import get_patient_context

    patient = await get_patient_context(patient_id)
    labs = patient.get("labs", [])

    egfr = _extract_lab_value(labs, "egfr", "gfr", "glomerular filtration")
    creatinine = _extract_lab_value(labs, "creatinine", "cr ", "scr")
    age = patient.get("age")
    weight_kg = patient.get("weight_kg")
    sex = patient.get("sex", "")

    crcl: float | None = None
    crcl_formula = ""
    if all(v is not None for v in (age, weight_kg, creatinine)) and creatinine > 0:
        crcl = _cockcroft_gault(int(age), float(weight_kg), float(creatinine), sex)
        sex_factor = "× 0.85 (female)" if sex and sex.lower() in ("f", "female") else ""
        crcl_formula = (
            f"[(140 − {age}) × {weight_kg}] ÷ [72 × {creatinine}] {sex_factor} = {crcl} mL/min"
        )

    renal_fn = egfr or crcl
    if renal_fn is None:
        md = (
            "## Renal Dose Review\n\n"
            "⚠️ Could not determine renal function — no eGFR or creatinine found in patient labs. "
            "Add lab values to the patient record to enable automated renal dose review.\n\n"
            "---\n⚠️ *Decision support only. Not a substitute for clinical judgment.*"
        )
        return {"markdown": md, "structured": {"error": "no_renal_function_data"}}

    ckd = _ckd_stage(float(renal_fn))

    if override_medications:
        meds = [m.strip().lower() for m in override_medications if m.strip()]
    else:
        raw_meds = patient.get("medications", [])
        meds = [
            (m["name"].lower() if isinstance(m, dict) else str(m).lower())
            for m in raw_meds
        ]

    if not meds:
        md = (
            "## Renal Dose Review\n\n"
            "No medications found in patient record.\n\n"
            "---\n⚠️ *Decision support only. Not a substitute for clinical judgment.*"
        )
        return {"markdown": md, "structured": {"egfr": egfr, "crcl": crcl, "medications": []}}

    # Fetch FDA dosing labels for all meds concurrently
    async with httpx.AsyncClient(timeout=15) as client:
        label_results = await asyncio.gather(
            *[_fetch_dosing_label(m, client) for m in meds],
            return_exceptions=True,
        )
    drug_labels = {
        drug: (r if isinstance(r, (str, type(None))) else None)
        for drug, r in zip(meds, label_results)
    }

    # Build Gemini prompt with all labels
    label_blocks = "\n\n".join(
        f"--- {drug.upper()} ---\n{text}" if text else f"--- {drug.upper()} ---\n[FDA label not found]"
        for drug, text in drug_labels.items()
    )

    renal_label = f"eGFR {egfr} mL/min/1.73m²" if egfr else ""
    crcl_label = f"CrCl {crcl} mL/min (Cockcroft-Gault)" if crcl else ""
    renal_summary = " | ".join(filter(None, [renal_label, crcl_label]))

    user_prompt = (
        f"Patient renal function: {renal_summary}\n"
        f"CKD stage: {ckd}\n\n"
        f"Medications to review: {', '.join(meds)}\n\n"
        f"FDA label sections:\n\n{label_blocks}\n\n"
        "Return JSON only."
    )

    resp = await _CLIENT.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_RENAL_SYSTEM,
            max_output_tokens=1024,
        ),
    )

    result = _parse_json(resp.text)
    med_results = result.get("medications", [])

    # Ensure every med is represented
    reviewed = {m["drug"].lower() for m in med_results}
    for m in meds:
        if m not in reviewed:
            med_results.append({
                "drug": m,
                "action": "MONITOR",
                "recommendation": "Review current prescribing information; automated review incomplete.",
                "threshold": "unknown",
                "label_found": drug_labels.get(m) is not None,
            })

    markdown = _to_markdown(egfr, crcl, ckd, crcl_formula, med_results)

    return {
        "markdown": markdown,
        "structured": {
            "egfr": egfr,
            "crcl": crcl,
            "ckd_stage": ckd,
            "medications": med_results,
        },
    }
