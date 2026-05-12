import xml.etree.ElementTree as ET

import httpx
from fastmcp.tools import tool

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


@tool()
async def search_arxiv(clinical_question: str, max_results: int = 5) -> list[dict]:
    """Search ArXiv for preprints and research papers relevant to a clinical question.

    Returns a list of papers with title, abstract, arxiv_id, year, authors,
    and a direct ArXiv URL. Pass the results to synthesize_evidence.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            _ARXIV_API,
            params={
                "search_query": f"all:{clinical_question}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        resp.raise_for_status()

    return _parse_arxiv_xml(resp.text)


def _parse_arxiv_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", _NS):
        raw_id = _ns_text(entry, "atom:id", _NS) or ""
        arxiv_id = raw_id.split("/abs/")[-1].strip()

        title = _ns_text(entry, "atom:title", _NS)
        summary = _ns_text(entry, "atom:summary", _NS)
        published = _ns_text(entry, "atom:published", _NS)
        year = published[:4] if published else None

        authors = [
            _ns_text(a, "atom:name", _NS)
            for a in entry.findall("atom:author", _NS)[:3]
        ]
        authors = [a for a in authors if a]

        if title and arxiv_id:
            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title.strip(),
                    "abstract": summary.strip() if summary else "No abstract available.",
                    "year": year,
                    "authors": authors,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                }
            )

    return papers


def _ns_text(element: ET.Element, xpath: str, ns: dict) -> str | None:
    node = element.find(xpath, ns)
    return node.text if node is not None else None
