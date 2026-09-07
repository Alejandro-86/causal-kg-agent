"""Fetch real biomedical abstracts from PubMed via NCBI E-utilities.

Free, no API key required (a key only raises the rate limit from 3 to
10 req/s, irrelevant at this corpus size).
"""

import time
import xml.etree.ElementTree as ET

import requests

from causal_kg.models import Abstract

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SEARCH_TERMS = [
    "GLP-1 receptor agonist mechanism",
    "semaglutide cardiovascular outcomes",
    "GLP-1 receptor agonist weight loss mechanism",
    "tirzepatide gastric emptying",
    "GLP-1 receptor agonist insulin secretion",
]


def _search_pmids(term: str, retmax: int = 8) -> list[str]:
    resp = requests.get(
        ESEARCH_URL,
        params={"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def _fetch_abstracts(pmids: list[str]) -> list[Abstract]:
    if not pmids:
        return []
    resp = requests.get(
        EFETCH_URL,
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    abstracts = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")
        abstract_texts = article.findall(".//AbstractText")

        if pmid_el is None or title_el is None or not abstract_texts:
            continue

        text = " ".join(
            (node.text or "") for node in abstract_texts
        ).strip()
        if not text:
            continue

        abstracts.append(
            Abstract(
                pmid=pmid_el.text,
                title="".join(title_el.itertext()),
                text=text,
                journal=journal_el.text if journal_el is not None else None,
                year=int(year_el.text) if year_el is not None and year_el.text.isdigit() else None,
            )
        )
    return abstracts


def fetch_glp1_abstracts(per_term: int = 8) -> list[Abstract]:
    """Fetch and dedupe abstracts across all search terms."""
    seen: dict[str, Abstract] = {}
    for term in SEARCH_TERMS:
        pmids = _search_pmids(term, retmax=per_term)
        for abstract in _fetch_abstracts(pmids):
            seen[abstract.pmid] = abstract
        time.sleep(0.4)  # be polite to the free E-utilities endpoint
    return list(seen.values())
