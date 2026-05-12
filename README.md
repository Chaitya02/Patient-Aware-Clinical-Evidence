# Patient-Aware Clinical Evidence

An MCP server for [Prompt Opinion](https://promptopinion.com) that answers clinical questions using the **patient's actual FHIR data** combined with **real peer-reviewed research** — not generic AI advice.

Built with [FastMCP](https://gofastmcp.com) and powered by Google Gemini, PubMed, and the OpenFDA API.

## Architecture

<img width="721" height="759" alt="Screenshot 2026-05-11 at 10 08 38 PM" src="https://github.com/user-attachments/assets/653d8ed9-5044-4361-b97b-3fd07a187890" />

The Gemini orchestrator routes each clinical question to the right specialist agent via A2A protocol. Each agent calls atomic MCP tools that fetch live patient data and real published evidence before synthesizing a patient-specific answer.

## What it does

| Tool | What it calls |
|---|---|
| `answer_clinical_question` | FHIR patient profile + PubMed search + Gemini synthesis |
| `check_drug_interactions` | FDA label DDI check (bidirectional) for a proposed drug |
| `check_renal_dosing` | Cockcroft-Gault CrCl + FDA label renal review for all patient meds |
| `get_patient_context` | Demographics, conditions, medications, labs, allergies from FHIR |
| `search_pubmed` | PubMed E-utilities search with study design and retraction status |
| `fetch_fda_label` | OpenFDA prescribing label sections for any drug |
| `synthesize_evidence` | Cited, graded answer from a list of articles (Gemini) |
| `search_arxiv` | Preprint search for cutting-edge research |

**Example — what makes this different from generic AI:**

> Generic AI: *"For atrial fibrillation, apixaban 5 mg twice daily is typical."*
>
> This tool: *"Jane Doe weighs 52 kg and has a creatinine of 1.8 mg/dL — she meets 2 of 3 dose-reduction criteria, so apixaban **2.5 mg BID** is correct. Her amiodarone (P-gp inhibitor) further supports this choice. \[PMID 38033089\]"*

The answer changes based on the actual patient. That is the point.

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```shell
git clone https://github.com/Chaitya02/Patient-Aware-Clinical-Evidence.git
cd Patient-Aware-Clinical-Evidence
uv sync
```

Copy `.env.example` to `.env` and add your Google Gemini API key:

```shell
GEMINI_API_KEY=your_key_here
```

Start the MCP server:

```shell
uv run python main.py
```

The server listens at `http://127.0.0.1:9000/mcp`.

To expose it publicly via ngrok:

```shell
ngrok http 9000
```

## FHIR Context (SHARP)

Prompt Opinion forwards patient context to every tool call via HTTP headers:

| Header | Description |
|---|---|
| `x-fhir-server-url` | Base URL of the FHIR server |
| `x-fhir-access-token` | Bearer token for the FHIR server |
| `x-patient-id` | Active patient ID |

When these headers are absent (e.g., local testing), the server falls back to the bundled synthetic demo patient (`jane_doe_fhir_bundle.json`).

## Demo Patient — Jane Doe (Synthetic)

A clinically complex 76-year-old designed to produce meaningfully different answers than generic AI:

- **Conditions:** New-onset Atrial Fibrillation, CKD Stage 3b (eGFR 38), Type 2 Diabetes, Hypertension
- **Medications:** Metformin, Amiodarone, Lisinopril, Atorvastatin
- **Labs:** eGFR 38, Creatinine 1.8 mg/dL, HbA1c 7.8%, INR 1.1
- **Allergies:** Penicillin (anaphylaxis), Sulfa drugs (rash)

## Project Structure

```
main.py                    # Server entry point — declares FHIR scopes, registers tools
po_fastmcp/                # Reusable FHIR context + FastMCP wrapper
tools/
  fhir_patient.py          # Fetches patient profile from FHIR
  clinical_qa.py           # answer_clinical_question — PICO + PubMed + synthesis
  drug_interactions.py     # check_drug_interactions — bidirectional FDA DDI check
  renal_dosing.py          # check_renal_dosing — Cockcroft-Gault + FDA label review
  pubmed_search.py         # search_pubmed — PubMed E-utilities
  fda_label.py             # fetch_fda_label — OpenFDA prescribing info
  synthesize.py            # synthesize_evidence — standalone Gemini synthesis
  arxiv_search.py          # search_arxiv — preprint search
jane_doe_fhir_bundle.json  # Synthetic demo patient (FHIR R4 bundle)
```

## Tech Stack

- **[FastMCP](https://gofastmcp.com)** — MCP server framework
- **[FHIR R4](https://hl7.org/fhir/R4/)** via `fhir.resources` + `httpx`
- **[Google Gemini](https://ai.google.dev)** (`gemini-3.1-flash-lite-preview`) — query reformulation, synthesis, DDI extraction
- **[PubMed E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25499/)** — free real-time literature search
- **[OpenFDA API](https://open.fda.gov/apis/drug/label/)** — FDA prescribing labels
- **[Prompt Opinion SHARP](https://promptopinion.com)** — patient context injection

## License

MIT
