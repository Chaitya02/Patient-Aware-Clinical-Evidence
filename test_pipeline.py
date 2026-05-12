"""
Run the full medical literature Q&A pipeline end-to-end and print every step.
Usage: uv run python test_pipeline.py
"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

from tools.arxiv_search import _ARXIV_API, _parse_arxiv_xml
from tools.pubmed_search import _EFETCH, _ESEARCH, _parse_pubmed_xml
from tools.synthesize import synthesize_evidence

QUESTION = "What is the evidence for SGLT2 inhibitors in heart failure?"
SEARCH_QUERY = "SGLT2 inhibitors heart failure"  # short keyword form for APIs


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def dump(label: str, data: object) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(data, indent=2, default=str))


async def main() -> None:
    banner("CLINICAL QUESTION")
    print(f"\n  {QUESTION}\n")

    # ── Step 1: PubMed search ─────────────────────────────────────
    banner("STEP 1 — PubMed Search")
    pubmed_params = {
        "db": "pubmed",
        "term": SEARCH_QUERY,
        "retmax": 3,
        "retmode": "json",
        "sort": "relevance",
    }
    print("\nRequest →")
    dump("GET esearch params", pubmed_params)

    async with httpx.AsyncClient(timeout=30) as client:
        search = await client.get(_ESEARCH, params=pubmed_params)
        pmids = search.json()["esearchresult"]["idlist"]
        print(f"\nFound PMIDs: {pmids}")

        fetch = await client.get(
            _EFETCH,
            params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        )

    pubmed_articles = _parse_pubmed_xml(fetch.text)
    print(f"\nParsed {len(pubmed_articles)} articles ↓")
    for a in pubmed_articles:
        print(f"  [{a['pmid']}] ({a['year']}) {a['title'][:70]}")

    # ── Step 2: ArXiv search ──────────────────────────────────────
    banner("STEP 2 — ArXiv Search")
    arxiv_params = {
        "search_query": f"all:{SEARCH_QUERY}",
        "start": 0,
        "max_results": 3,
        "sortBy": "relevance",
    }
    print("\nRequest →")
    dump("GET arxiv params", arxiv_params)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(_ARXIV_API, params=arxiv_params)

    arxiv_papers = _parse_arxiv_xml(resp.text)
    print(f"\nParsed {len(arxiv_papers)} papers ↓")
    for p in arxiv_papers:
        print(f"  [{p['arxiv_id']}] ({p['year']}) {p['title'][:70]}")

    # ── Step 3: Combine ───────────────────────────────────────────
    banner("STEP 3 — Combined Input to Synthesize")
    all_articles = pubmed_articles + arxiv_papers
    print(f"\nTotal articles passed to synthesize_evidence: {len(all_articles)}")
    dump("articles (input)", all_articles)

    # ── Step 4: Synthesize ────────────────────────────────────────
    banner("STEP 4 — Synthesis (Gemini 2.5 Flash)")
    print(f"\nModel: gemini-2.5-flash")
    print(f"Question: {QUESTION}")
    print("\nCalling synthesize_evidence …")

    result = await synthesize_evidence(QUESTION, all_articles)

    banner("FINAL OUTPUT")
    dump("synthesize_evidence result", result)

    print(f"\n  Answer       : {result['answer']}")
    print(f"  Evidence grade: {result['evidence_grade'].upper()}")
    print(f"  Rationale    : {result['grade_rationale']}")
    print(f"\n  Citations:")
    for c in result.get("citations", []):
        print(f"    [{c['number']}] {c['title']} ({c.get('year','')}) — {c.get('url','')}")
    print()


asyncio.run(main())
