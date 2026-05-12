import httpx
from fastmcp.tools import tool

_OPENFDA = "https://api.fda.gov/drug/label.json"

_SECTIONS = {
    "drug_interactions",
    "dosage_and_administration",
    "warnings_and_precautions",
    "boxed_warning",
    "contraindications",
    "indications_and_usage",
}


@tool()
async def fetch_fda_label(
    drug_name: str,
    sections: list[str] | None = None,
) -> dict:
    """Fetch FDA prescribing label sections for a drug from OpenFDA.

    Returns the requested label sections as plain text. Useful for drug interaction
    text, renal dosing tables, contraindications, and boxed warnings.

    Args:
        drug_name: Generic or brand name of the drug.
        sections: Label sections to return. Defaults to all available sections.
                  Valid values: drug_interactions, dosage_and_administration,
                  warnings_and_precautions, boxed_warning, contraindications,
                  indications_and_usage.
    """
    requested = set(sections) & _SECTIONS if sections else _SECTIONS

    async with httpx.AsyncClient(timeout=15) as client:
        for search_field in ("openfda.generic_name", "openfda.brand_name"):
            try:
                resp = await client.get(
                    _OPENFDA,
                    params={"search": f'{search_field}:"{drug_name}"', "limit": 1},
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if not results:
                        continue
                    r = results[0]
                    openfda = r.get("openfda", {})
                    found: dict[str, str] = {}
                    for section in requested:
                        data = r.get(section, [])
                        if data:
                            found[section] = data[0]
                    return {
                        "drug_name": drug_name,
                        "generic_name": openfda.get("generic_name", [drug_name])[0],
                        "brand_names": openfda.get("brand_name", []),
                        "sections_found": list(found.keys()),
                        "sections_missing": [s for s in requested if s not in found],
                        "label": found,
                    }
            except Exception:
                continue

    return {
        "drug_name": drug_name,
        "generic_name": drug_name,
        "brand_names": [],
        "sections_found": [],
        "sections_missing": list(requested),
        "label": {},
        "error": "FDA label not found for this drug name.",
    }
