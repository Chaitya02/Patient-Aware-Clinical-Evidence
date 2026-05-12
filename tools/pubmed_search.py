import xml.etree.ElementTree as ET

import httpx
from fastmcp.tools import tool

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Maps PubMed PublicationType strings to (evidence_tier, canonical_label).
# Lower tier = higher quality. Used downstream for GRADE computation.
_STUDY_DESIGN_TIERS: dict[str, tuple[int, str]] = {
    "meta-analysis": (1, "Meta-Analysis"),
    "systematic review": (1, "Systematic Review"),
    "randomized controlled trial": (2, "RCT"),
    "controlled clinical trial": (2, "Controlled Clinical Trial"),
    "clinical trial, phase iii": (2, "Phase III RCT"),
    "clinical trial, phase iv": (2, "Phase IV RCT"),
    "clinical trial": (3, "Clinical Trial"),
    "multicenter study": (3, "Multicenter Study"),
    "observational study": (3, "Observational Study"),
    "comparative study": (3, "Comparative Study"),
    "validation study": (3, "Validation Study"),
    "review": (4, "Review"),
    "practice guideline": (4, "Practice Guideline"),
    "guideline": (4, "Guideline"),
    "consensus development conference": (4, "Consensus Statement"),
    "journal article": (4, "Journal Article"),
    "case reports": (5, "Case Report"),
    "twin study": (5, "Twin Study"),
    "letter": (6, "Letter"),
    "editorial": (6, "Editorial"),
    "comment": (6, "Comment"),
    "personal narrative": (6, "Personal Narrative"),
    "news": (6, "News"),
}


def _classify_study_design(pub_types: list[str]) -> tuple[int, str]:
    """Return (evidence_tier, label) for the highest-quality design in pub_types."""
    best_tier, best_label = 99, "Unclassified"
    for pt in pub_types:
        tier, label = _STUDY_DESIGN_TIERS.get(pt.lower(), (99, ""))
        if tier < best_tier:
            best_tier, best_label = tier, label
    if best_tier == 99:
        best_tier, best_label = 4, "Journal Article"
    return best_tier, best_label


@tool()
async def search_pubmed(clinical_question: str, max_results: int = 5) -> list[dict]:
    """Search PubMed for peer-reviewed evidence on a clinical question.

    Returns articles enriched with retraction status, study design classification,
    and evidence tier. Retracted articles are flagged but still returned so callers
    can decide whether to exclude them.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        search = await client.get(
            _ESEARCH,
            params={
                "db": "pubmed",
                "term": clinical_question,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        search.raise_for_status()
        pmids: list[str] = search.json()["esearchresult"]["idlist"]

        if not pmids:
            return []

        fetch = await client.get(
            _EFETCH,
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            },
        )
        fetch.raise_for_status()

    return _parse_pubmed_xml(fetch.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article, ".//PMID")
        title = _text(article, ".//ArticleTitle")
        abstract = _text(article, ".//AbstractText")
        year = _text(article, ".//PubDate/Year") or _text(article, ".//PubDate/MedlineDate")
        journal = _text(article, ".//Journal/Title")

        authors = []
        for author in article.findall(".//Author")[:3]:
            last = _text(author, "LastName")
            first = _text(author, "ForeName")
            if last:
                initials = f" {first[0]}" if first else ""
                authors.append(f"{last}{initials}")

        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]

        # Retraction check via PubMed's own CommentsCorrections links.
        # RefType "RetractionIn" means this article has been retracted.
        # RefType "ExpressionOfConcernIn" means a concern notice exists.
        retracted = False
        retraction_notice_pmid: str | None = None
        expression_of_concern = False
        for correction in article.findall(".//CommentsCorrections"):
            ref_type = correction.get("RefType", "")
            if ref_type == "RetractionIn":
                retracted = True
                retraction_notice_pmid = _text(correction, "PMID")
            elif ref_type == "ExpressionOfConcernIn":
                expression_of_concern = True

        # DOI (used by downstream tools for cross-referencing)
        doi: str | None = None
        for loc in article.findall(".//ELocationID"):
            if loc.get("EIdType") == "doi" and loc.text:
                doi = loc.text
                break

        study_tier, study_design = _classify_study_design(pub_types)

        if title:
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract or "No abstract available.",
                    "year": year,
                    "journal": journal,
                    "authors": authors,
                    "pub_types": pub_types,
                    "study_design": study_design,
                    "study_tier": study_tier,
                    "retracted": retracted,
                    "retraction_notice_pmid": retraction_notice_pmid,
                    "expression_of_concern": expression_of_concern,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

    return articles


def _text(element: ET.Element, xpath: str) -> str | None:
    node = element.find(xpath)
    return node.text if node is not None else None
