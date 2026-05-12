import json
import re

from google import genai
from google.genai import types
from fastmcp.tools import tool

_MODEL = "gemini-3.1-flash-lite-preview"

_SYSTEM = """You are a clinical evidence synthesizer. Given a clinical question and a set of abstracts, return ONLY valid JSON — no markdown, no preamble, no code fences:

{
  "recommendation": "one concise sentence clinical recommendation",
  "evidence_summary": "3-4 sentences summarising the body of evidence",
  "evidence_grade": "HIGH|MODERATE|LOW|INSUFFICIENT",
  "grade_rationale": "one sentence explaining the grade",
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

GRADE definitions:
- HIGH: multiple consistent RCTs or systematic reviews
- MODERATE: some RCTs or consistent observational studies
- LOW: case reports, expert opinion, or inconsistent findings
- INSUFFICIENT: no relevant evidence found"""

_USER_TEMPLATE = """Clinical question: {question}

Literature ({n} sources):
{articles}

Return JSON only."""


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
            "evidence_summary": "",
            "evidence_grade": grade_match.group(1).upper() if grade_match else "LOW",
            "grade_rationale": "Response was truncated.",
            "citations": [],
        }
    raise ValueError(f"Could not parse JSON from model response: {text[:300]}")


def _to_markdown(r: dict) -> str:
    lines: list[str] = []

    lines += ["## Recommendation", f"**{r.get('recommendation', '')}**", ""]
    lines += ["## Evidence Summary", r.get("evidence_summary", ""), ""]

    grade = r.get("evidence_grade", "")
    rationale = r.get("grade_rationale", "")
    lines += ["## Evidence Grade", f"**{grade}** — {rationale}", ""]

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
async def synthesize_evidence(
    clinical_question: str,
    articles: list[dict],
) -> dict:
    """Synthesize PubMed and ArXiv search results into a cited clinical answer.

    Takes the combined output of search_pubmed and search_arxiv, calls Gemini
    to produce a concise evidence-based answer with inline citations and a
    GRADE evidence level.

    Returns both a `markdown` string (rendered in chat) and a `structured` dict
    (for downstream agents). Requires GEMINI_API_KEY in the environment.
    """
    if not articles:
        md = (
            "## Evidence Grade\n**INSUFFICIENT**\n\n"
            "No articles were provided for synthesis.\n\n"
            "---\n⚠️ *Decision support only. Not a substitute for clinical judgment.*"
        )
        return {
            "markdown": md,
            "structured": {
                "recommendation": None,
                "evidence_grade": "INSUFFICIENT",
                "citations": [],
            },
        }

    articles_text = "\n\n".join(
        f"[{i + 1}] {a.get('title', 'Unknown title')} "
        f"({a.get('year') or 'n/a'}) — "
        f"{a.get('journal') or 'ArXiv'}\n"
        f"ID: {a.get('pmid') or a.get('arxiv_id', 'n/a')}\n"
        f"Abstract: {(a.get('abstract') or '')[:600]}"
        for i, a in enumerate(articles)
    )

    user_prompt = _USER_TEMPLATE.format(
        question=clinical_question,
        n=len(articles),
        articles=articles_text,
    )

    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=2048,
        ),
    )

    result = _parse_json(response.text)

    for c in result.get("citations", []):
        pmid = c.get("pmid", "")
        if pmid:
            c["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    markdown = _to_markdown(result)

    return {
        "markdown": markdown,
        "structured": {
            "recommendation": result.get("recommendation"),
            "evidence_summary": result.get("evidence_summary"),
            "evidence_grade": result.get("evidence_grade"),
            "grade_rationale": result.get("grade_rationale"),
            "citations": result.get("citations", []),
        },
    }
