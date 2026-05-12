import json
import re

from google import genai
from google.genai import types
from fastmcp.tools import tool

_CLIENT = genai.Client()
_MODEL = "gemini-3.1-flash-lite-preview"


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        chunk = text[start:]
        match = re.search(r"\{[\s\S]*\}", chunk)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        grade_match = re.search(r'"evidence_grade"\s*:\s*"(\w+)"', chunk)
        return {
            "recommendation": text,
            "patient_reasoning": [],
            "evidence_summary": "",
            "evidence_grade": grade_match.group(1).upper() if grade_match else "LOW",
            "grade_rationale": "Response was truncated; evidence grade may be incomplete.",
            "verify_independently": [],
            "citations": [],
        }
    raise ValueError(f"Could not parse JSON from model response: {text[:300]}")


def _classify_evidence(articles: list[dict]) -> tuple[str, str]:
    """Classify retrieved evidence using study_tier from pubmed_search.

    Tiers (set in pubmed_search._classify_study_design):
      1 = Meta-Analysis / Systematic Review
      2 = RCT / Controlled Trial
      3 = Observational / Multicenter
      4 = Review / Guideline / Journal Article
      5 = Case Report
      6 = Letter / Editorial / Opinion

    Returns (bucket, human_label):
      "high"         — min tier ≤ 2 (RCT or better)
      "mixed"        — min tier 3–4 (observational / review)
      "case_reports" — all articles tier 5–6
      "none"         — no articles
    """
    if not articles:
        return "none", "no articles"

    tiers = [a.get("study_tier", 4) for a in articles]
    min_tier = min(tiers)

    if min_tier <= 2:
        best = next(a for a in articles if a.get("study_tier", 99) <= 2)
        return "high", f"includes {best.get('study_design', 'controlled trial')}"
    if min_tier <= 4:
        designs = sorted({a.get("study_design", "Journal Article") for a in articles})
        return "mixed", f"observational/review evidence ({', '.join(designs)})"
    return "case_reports", "case reports / opinion only"


def _retraction_warning_lines(retracted: list[dict]) -> list[str]:
    if not retracted:
        return []
    lines = [
        "## 🚫 Retracted Articles Excluded",
        "",
        f"**{len(retracted)} article(s) were automatically excluded** from synthesis "
        "because PubMed has marked them as retracted.",
        "",
    ]
    for a in retracted:
        notice = (
            f" ([retraction notice](https://pubmed.ncbi.nlm.nih.gov/{a['retraction_notice_pmid']}/)"
            f")" if a.get("retraction_notice_pmid") else ""
        )
        lines.append(f"- PMID {a['pmid']}: *{a['title']}*{notice}")
    lines.append("")
    return lines


def _insufficient_markdown(reason: str, search_query: str, articles: list[dict]) -> str:
    lines = [
        "## ⚠️ INSUFFICIENT EVIDENCE",
        "",
        f"**{reason.upper()}**",
        "",
        "No peer-reviewed controlled trials, RCTs, or systematic reviews were found. "
        "A clinical recommendation cannot be responsibly synthesised from the available evidence.",
        "",
    ]

    if articles:
        lines += ["**Articles retrieved (not synthesised):**", ""]
        for i, a in enumerate(articles, 1):
            design_label = a.get("study_design", ", ".join(a.get("pub_types", [])) or "type unknown")
            lines.append(
                f"[{i}] {a.get('authors', [''])[0] if a.get('authors') else ''} "
                f"*{a['title']}.* {a.get('journal', '')}, {a.get('year', '')} "
                f"— _{design_label}_  \n    {a['url']}"
            )
        lines.append("")

    lines += [
        f"**PubMed query used:** `{search_query}`",
        "",
        "---",
        "⚠️ *Decision support only. Consult clinical guidelines or a specialist.*",
    ]
    return "\n".join(lines)


_PICO_SYSTEM = (
    "You are a clinical librarian. Convert the clinical question + patient context "
    "into a concise PubMed MeSH-style search query (max 8 terms, AND/OR operators). "
    "Return ONLY the query string — no explanation, no quotes."
)

_SYNTHESIS_SYSTEM = """You are a clinical evidence synthesizer providing patient-specific answers.

Given a clinical question, the patient's profile, and relevant abstracts, return ONLY valid JSON with this exact schema — no markdown, no preamble, no code fences:

{
  "recommendation": "one concise sentence stating the patient-specific clinical recommendation",
  "patient_reasoning": [
    {
      "factor": "specific patient data point (e.g. Weight 52 kg + Creatinine 1.8 mg/dL)",
      "reasoning": "clinical implication for this patient",
      "citation_numbers": [1]
    }
  ],
  "evidence_summary": "3-4 sentences summarising the body of evidence and monitoring requirements",
  "evidence_grade": "HIGH|MODERATE|LOW|INSUFFICIENT",
  "grade_rationale": "one sentence explaining the grade",
  "conflicts": [
    {
      "paper_a": 1,
      "paper_b": 2,
      "note": "one sentence describing the disagreement and a plausible explanation (e.g. different dosing, population, follow-up duration)"
    }
  ],
  "verify_independently": [
    "specific item the clinician must confirm at the point of care"
  ],
  "citations": [
    {
      "number": 1,
      "pmid": "12345678",
      "authors": "Last F et al.",
      "title": "Full paper title",
      "journal": "Journal Name",
      "year": "2021"
    }
  ]
}

Rules:
- citation_numbers in patient_reasoning must reference valid numbers in the citations array
- evidence_grade must be exactly one of: HIGH, MODERATE, LOW, INSUFFICIENT (uppercase)
- Base evidence_grade on the study design tiers provided with each source:
    HIGH       → best available source is Tier 1 (Meta-Analysis/SR) or Tier 2 (RCT)
    MODERATE   → best available source is Tier 3 (Observational/Clinical Trial)
    LOW        → best available source is Tier 4 (Review/Guideline/Journal Article)
    INSUFFICIENT → all sources are Tier 5–6 (Case Report/Opinion)
- Include 3-6 patient_reasoning bullet points, one per clinically relevant patient characteristic
- Include every source used; minimum 2 citations
- conflicts: compare papers on outcomes, effect sizes, safety rates, or recommendations. Include a conflict only when papers genuinely disagree — omit the field (or use []) if the literature is consistent"""


def _patient_summary(patient: dict) -> str:
    def _fmt(items: list, key: str | None = None) -> str:
        if not items:
            return "none documented"
        if key:
            return ", ".join(
                item[key] if isinstance(item, dict) else str(item) for item in items
            )
        return ", ".join(str(i) for i in items)

    labs = ", ".join(
        f"{l['test']} {l['value']}{l.get('unit','')} [{l.get('flag','normal')}]"
        if isinstance(l, dict) else str(l)
        for l in patient.get("labs", [])
    ) or "none"

    return (
        f"Age: {patient.get('age')} | Sex: {patient.get('sex')} | "
        f"Weight: {patient.get('weight_kg')} kg\n"
        f"Active conditions: {_fmt(patient.get('conditions', []), 'display')}\n"
        f"Current medications: {_fmt(patient.get('medications', []), 'name')}\n"
        f"Recent labs: {labs}\n"
        f"Allergies: {_fmt(patient.get('allergies', []))}"
    )


def _to_markdown(r: dict, retracted: list[dict] | None = None) -> str:
    lines: list[str] = []

    if retracted:
        lines += _retraction_warning_lines(retracted)

    lines += ["## Recommendation", f"**{r.get('recommendation', '')}**", ""]

    lines.append("## Patient-Specific Reasoning")
    for item in r.get("patient_reasoning", []):
        nums = "".join(f"[{n}]" for n in item.get("citation_numbers", []))
        suffix = f" {nums}" if nums else ""
        lines.append(f"- **{item['factor']}** → {item['reasoning']}{suffix}")
    lines.append("")

    lines += ["## Evidence Summary", r.get("evidence_summary", ""), ""]

    grade = r.get("evidence_grade", "")
    rationale = r.get("grade_rationale", "")
    lines += ["## Evidence Grade", f"**{grade}** — {rationale}", ""]

    conflicts = r.get("conflicts", [])
    if conflicts:
        lines.append("## Evidence Conflicts")
        for c in conflicts:
            lines.append(f"> **[{c['paper_a']}] vs [{c['paper_b']}]:** {c['note']}")
        lines.append("")

    verify = r.get("verify_independently", [])
    if verify:
        lines.append("## What to Verify Independently")
        lines += [f"- {v}" for v in verify]
        lines.append("")

    lines.append("## References")
    for c in r.get("citations", []):
        pmid = c.get("pmid", "")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else c.get("url", "")
        authors = c.get("authors", "")
        title = c.get("title", "")
        journal = c.get("journal", "")
        year = c.get("year", "")
        meta = " ".join(filter(None, [authors, f"*{title}.*" if title else None, journal, year]))
        lines.append(f"[{c['number']}] {meta}  \n    {url}")
    lines.append("")

    lines += ["---", "⚠️ *Decision support only. Not a substitute for clinical judgment.*"]
    return "\n".join(lines)


@tool()
async def answer_clinical_question(
    question: str,
    patient_id: str | None = None,
    max_sources: int = 5,
) -> dict:
    """Answer a clinical question grounded in the patient's context and peer-reviewed evidence.

    This is the primary clinical decision-support tool. It:
    1. Retrieves the patient's clinical profile (demographics, conditions, medications, labs, allergies).
    2. Reformulates the question as a PICO-informed PubMed search query tailored to this patient.
    3. Retrieves up to max_sources peer-reviewed articles from PubMed.
    4. Synthesizes a patient-specific, cited answer with a GRADE evidence grade.

    Returns both a `markdown` string (rendered in chat) and a `structured` dict (for downstream agents).
    Every citation is a clickable PubMed URL. Always ends with a clinical judgment disclaimer.
    Call this instead of search_pubmed + synthesize_evidence for any patient-facing question.
    """
    from tools.fhir_patient import get_patient_context
    from tools.pubmed_search import search_pubmed

    patient = await get_patient_context(patient_id)
    summary = _patient_summary(patient)

    # Step 1: PICO-informed query reformulation
    pico_resp = await _CLIENT.aio.models.generate_content(
        model=_MODEL,
        contents=f"Clinical question: {question}\n\nPatient:\n{summary}\n\nPubMed query:",
        config=types.GenerateContentConfig(
            system_instruction=_PICO_SYSTEM,
            max_output_tokens=150,
        ),
    )
    search_query = pico_resp.text.strip()

    # Step 2: PubMed retrieval
    all_articles = await search_pubmed(search_query, max_results=max_sources)

    # Step 3: Retraction filter — never synthesise from retracted articles
    retracted = [a for a in all_articles if a.get("retracted")]
    articles = [a for a in all_articles if not a.get("retracted")]

    # Step 4: Evidence quality gate — intercept before synthesis
    tier, tier_label = _classify_evidence(articles)

    if tier in ("none", "case_reports"):
        reason = (
            "No relevant peer-reviewed literature found"
            if tier == "none"
            else "INSUFFICIENT EVIDENCE — case reports only"
        )
        md = _insufficient_markdown(reason, search_query, articles)
        if retracted:
            md = "\n".join(_retraction_warning_lines(retracted)) + "\n" + md
        return {
            "markdown": md,
            "structured": {
                "recommendation": None,
                "evidence_grade": "INSUFFICIENT",
                "evidence_tier": tier,
                "retracted_excluded": len(retracted),
                "citations": [
                    {"pmid": a["pmid"], "title": a["title"], "url": a["url"],
                     "study_design": a.get("study_design")}
                    for a in articles
                ],
            },
            "patient_context": patient,
            "pico_query": search_query,
        }

    # Step 5: Patient-aware synthesis
    articles_text = "\n\n".join(
        f"[{i+1}] [{a.get('study_design', 'Journal Article')} | Tier {a.get('study_tier', 4)}] "
        f"{a['title']} ({a.get('year','n/a')}) — {a.get('journal','Unknown')}\n"
        f"PMID: {a['pmid']} | URL: {a['url']}\n"
        f"Abstract: {(a.get('abstract') or '')[:300]}"
        for i, a in enumerate(articles)
    )

    user_prompt = (
        f"Clinical question: {question}\n\n"
        f"Patient profile:\n{summary}\n\n"
        f"PubMed search query used: {search_query}\n\n"
        f"Evidence quality assessment: {tier_label}\n\n"
        f"Literature ({len(articles)} sources, with study design and tier):\n{articles_text}\n\n"
        "Return JSON only."
    )

    synthesis_resp = await _CLIENT.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYNTHESIS_SYSTEM,
            max_output_tokens=2048,
        ),
    )

    result = _parse_json(synthesis_resp.text)

    for c in result.get("citations", []):
        pmid = c.get("pmid", "")
        if pmid:
            c["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    markdown = _to_markdown(result, retracted=retracted)

    return {
        "markdown": markdown,
        "structured": {
            "recommendation": result.get("recommendation"),
            "patient_reasoning": result.get("patient_reasoning", []),
            "evidence_summary": result.get("evidence_summary"),
            "evidence_grade": result.get("evidence_grade"),
            "grade_rationale": result.get("grade_rationale"),
            "conflicts": result.get("conflicts", []),
            "verify_independently": result.get("verify_independently", []),
            "citations": result.get("citations", []),
            "retracted_excluded": [
                {"pmid": a["pmid"], "title": a["title"]} for a in retracted
            ],
        },
        "patient_context": patient,
        "pico_query": search_query,
    }
