from dotenv import load_dotenv

load_dotenv()

from po_fastmcp import POFastMCP
from tools import register_tools

mcp = POFastMCP(
    name="Patient-Aware Clinical Evidence",
    instructions=(
        "You are a clinical decision-support assistant. "
        "Always copy the `markdown` field from tool results VERBATIM — never rephrase or add text.\n\n"
        "TOOLS — two tiers:\n\n"
        "ATOMIC (one API call each):\n"
        "• get_patient_context — fetch FHIR patient profile (demographics, meds, labs, conditions)\n"
        "• search_pubmed — search PubMed; returns articles with study design, tier, retraction status\n"
        "• fetch_fda_label — fetch FDA prescribing label sections for any drug (OpenFDA)\n"
        "• synthesize_evidence — synthesize a list of articles into a cited answer (Gemini)\n\n"
        "SPECIALIST (compose atomic tools internally):\n"
        "• answer_clinical_question — full pipeline: PICO + PubMed + patient-aware synthesis\n"
        "• check_drug_interactions — bidirectional FDA label DDI check for a proposed medication\n"
        "• check_renal_dosing — Cockcroft-Gault + FDA label renal review for all patient meds\n\n"
        "ROUTING:\n"
        "• Clinical question → answer_clinical_question\n"
        "• Drug interactions / safe to add? → check_drug_interactions\n"
        "• Renal dose review → check_renal_dosing\n"
        "• Patient profile / labs / meds → get_patient_context\n"
        "• Raw PubMed search → search_pubmed\n"
        "• FDA label lookup → fetch_fda_label\n"
        "• Synthesize provided articles → synthesize_evidence\n\n"
        "NEVER answer from your own knowledge."
    ),
    fhir_scopes=[
        {"name": "patient/Patient.rs",              "required": True},
        {"name": "patient/Condition.rs"},
        {"name": "patient/MedicationStatement.rs"},
        {"name": "patient/Observation.rs"},
        {"name": "patient/AllergyIntolerance.rs"},
    ],
)

register_tools(mcp)


def main() -> None:
    try:
        print("Starting MCP server at http://127.0.0.1:9000/mcp")
        print("Press Ctrl+C to stop.")
        mcp.run(transport="http", host="127.0.0.1", port=9000)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
