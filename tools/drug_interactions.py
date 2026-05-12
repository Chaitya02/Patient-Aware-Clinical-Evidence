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

_SEVERITY_ORDER = {"MAJOR": 0, "MODERATE": 1, "MINOR": 2, "UNKNOWN": 3}

_EXTRACTION_SYSTEM = """You are a clinical pharmacist extracting drug-drug interactions from FDA prescribing information.

You are given FDA drug_interactions label text for MULTIPLE drugs (the proposed medication and each current patient medication). Check BOTH directions:
- Does the proposed drug's label mention any patient medication?
- Does any patient medication's label mention the proposed drug or its drug class?

Return ONLY valid JSON — no markdown, no preamble:

{
  "interactions": [
    {
      "drug_b": "exact name of the patient medication that interacts",
      "severity": "MAJOR|MODERATE|MINOR|UNKNOWN",
      "mechanism": "brief pharmacokinetic or pharmacodynamic mechanism",
      "effect": "clinical consequence if the combination is used",
      "management": "specific recommended action (dose adjustment, monitoring, avoid, etc.)",
      "source_direction": "proposed_label|patient_med_label|both"
    }
  ],
  "no_interactions_noted": ["drug names not mentioned in any interaction text"]
}

Severity:
- MAJOR: life-threatening potential; avoid or requires immediate dose adjustment
- MODERATE: clinically significant; requires monitoring or possible dose modification
- MINOR: minimal clinical effect; awareness only
- UNKNOWN: interaction mentioned but severity not characterised

Every patient medication must appear in exactly one of "interactions" or "no_interactions_noted"."""


async def _fetch_label_section(drug_name: str, client: httpx.AsyncClient, section: str = "drug_interactions") -> str | None:
    for search_field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            resp = await client.get(
                _OPENFDA,
                params={"search": f'{search_field}:"{drug_name}"', "limit": 1},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    data = results[0].get(section, [])
                    return data[0] if data else None
        except Exception:
            continue
    return None


async def _fetch_all_interaction_labels(drugs: list[str]) -> dict[str, str | None]:
    async with httpx.AsyncClient(timeout=15) as client:
        results = await asyncio.gather(
            *[_fetch_label_section(d, client, "drug_interactions") for d in drugs],
            return_exceptions=True,
        )
    return {
        drug: (r if isinstance(r, (str, type(None))) else None)
        for drug, r in zip(drugs, results)
    }


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
    return {"interactions": [], "no_interactions_noted": []}


def _severity_badge(severity: str) -> str:
    return {"MAJOR": "🔴 MAJOR", "MODERATE": "🟡 MODERATE", "MINOR": "🟢 MINOR"}.get(
        severity, "⚪ UNKNOWN"
    )


def _to_markdown(
    proposed: str,
    patient_meds: list[str],
    interactions: list[dict],
    no_interactions: list[str],
    labels_found: dict[str, bool],
) -> str:
    lines: list[str] = []

    lines += [
        "## Drug Interaction Check",
        f"**Proposed medication:** {proposed.title()}  ",
        f"**Checking against:** {', '.join(m.title() for m in patient_meds)}",
        f"**Data source:** FDA prescribing information (bidirectional check)",
        "",
    ]

    missing = [d for d, found in labels_found.items() if not found]
    if missing:
        lines += [
            f"⚠️ FDA label not found for: {', '.join(m.title() for m in missing)} — "
            "those drugs were excluded from automated review.",
            "",
        ]

    if interactions:
        lines.append("## ⚠️ Interactions Found")
        lines.append("")
        for ix in sorted(interactions, key=lambda x: _SEVERITY_ORDER.get(x.get("severity", "UNKNOWN"), 3)):
            badge = _severity_badge(ix.get("severity", "UNKNOWN"))
            lines += [
                f"### {proposed.title()} ↔ {ix['drug_b'].title()} — {badge}",
                f"- **Mechanism:** {ix.get('mechanism', 'Not specified')}",
                f"- **Effect:** {ix.get('effect', 'Not specified')}",
                f"- **Management:** {ix.get('management', 'Not specified')}",
                f"- *Source: FDA prescribing information*",
                "",
            ]
    else:
        lines += ["## ✓ No Interactions Found", ""]

    if no_interactions:
        lines += [
            "## ✓ No Significant Interactions Noted",
            ", ".join(d.title() for d in no_interactions),
            "",
        ]

    lines += [
        "---",
        "⚠️ *Decision support only. Not a substitute for clinical judgment. "
        "Consult a pharmacist or current interaction database before prescribing.*",
    ]
    return "\n".join(lines)


@tool()
async def check_drug_interactions(
    proposed_medication: str,
    patient_id: str | None = None,
    override_medications: list[str] | None = None,
) -> dict:
    """Check drug-drug interactions between a proposed medication and the patient's current medications.

    Fetches FDA prescribing labels for the proposed medication AND each patient medication
    concurrently, then checks both directions: does the proposed drug's label mention any
    patient med, and does any patient med's label mention the proposed drug?

    Data source: FDA drug label (OpenFDA). Returns both `markdown` (for chat) and
    `structured` (for downstream agents).

    Args:
        proposed_medication: Drug being considered (generic or brand name).
        patient_id: FHIR patient ID — used to pull the current medication list automatically.
        override_medications: Explicit med list; used instead of FHIR if provided.
    """
    if override_medications:
        patient_meds = [m.strip().lower() for m in override_medications if m.strip()]
    else:
        from tools.fhir_patient import get_patient_context
        patient = await get_patient_context(patient_id)
        meds = patient.get("medications", [])
        patient_meds = [
            (m["name"].lower() if isinstance(m, dict) else str(m).lower())
            for m in meds
        ]

    proposed = proposed_medication.strip().lower()
    check_meds = [m for m in patient_meds if m != proposed]

    if not check_meds:
        md = (
            "## Drug Interaction Check\n\n"
            "No other medications to check against.\n\n"
            "---\n⚠️ *Decision support only. Not a substitute for clinical judgment.*"
        )
        return {"markdown": md, "structured": {"interactions": [], "no_interactions_noted": []}}

    # Fetch all labels concurrently (proposed + every patient med)
    all_drugs = [proposed] + check_meds
    labels = await _fetch_all_interaction_labels(all_drugs)
    labels_found = {d: labels[d] is not None for d in all_drugs}

    # Build combined label context for bidirectional analysis
    label_sections = []
    for drug, text in labels.items():
        if text:
            direction = "PROPOSED MEDICATION" if drug == proposed else "PATIENT MEDICATION"
            label_sections.append(
                f"--- {direction}: {drug.upper()} ---\n{text[:2000]}"
            )

    if not label_sections:
        md = _to_markdown(proposed, check_meds, [], check_meds, labels_found)
        return {
            "markdown": md,
            "structured": {
                "proposed_medication": proposed,
                "labels_found": labels_found,
                "interactions": [],
                "no_interactions_noted": check_meds,
            },
        }

    user_prompt = (
        f"Proposed medication: {proposed}\n"
        f"Patient's current medications: {', '.join(check_meds)}\n\n"
        f"FDA drug_interactions label texts:\n\n"
        + "\n\n".join(label_sections)
        + "\n\nReturn JSON only."
    )

    resp = await _CLIENT.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_EXTRACTION_SYSTEM,
            max_output_tokens=1024,
        ),
    )

    result = _parse_json(resp.text)
    interactions = result.get("interactions", [])
    no_interactions = result.get("no_interactions_noted", [])

    accounted = {ix["drug_b"].lower() for ix in interactions} | {d.lower() for d in no_interactions}
    for m in check_meds:
        if m not in accounted:
            no_interactions.append(m)

    markdown = _to_markdown(proposed, check_meds, interactions, no_interactions, labels_found)

    return {
        "markdown": markdown,
        "structured": {
            "proposed_medication": proposed,
            "patient_medications": check_meds,
            "labels_found": labels_found,
            "interactions": interactions,
            "no_interactions_noted": no_interactions,
        },
    }
