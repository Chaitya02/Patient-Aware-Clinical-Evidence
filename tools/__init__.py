from fastmcp import FastMCP

from tools.clinical_qa import answer_clinical_question
from tools.drug_interactions import check_drug_interactions
from tools.fda_label import fetch_fda_label
from tools.fhir_patient import get_patient_context
from tools.pubmed_search import search_pubmed
from tools.renal_dosing import check_renal_dosing
from tools.synthesize import synthesize_evidence


def register_tools(mcp: FastMCP) -> None:
    # --- Atomic API tools (one per external service) ---
    mcp.add_tool(get_patient_context)       # FHIR
    mcp.add_tool(search_pubmed)             # PubMed eUtils
    mcp.add_tool(fetch_fda_label)           # OpenFDA
    mcp.add_tool(synthesize_evidence)       # Gemini synthesis

    # --- Specialist tools (compose the atomic ones) ---
    mcp.add_tool(answer_clinical_question)  # PICO + PubMed + Gemini
    mcp.add_tool(check_drug_interactions)   # OpenFDA DDI + Gemini
    mcp.add_tool(check_renal_dosing)        # OpenFDA + Gemini + Cockcroft-Gault
