"""LLM-driven extraction of CausalRelation objects from a single abstract.

Grounding rule: evidence_sentence must come from the abstract text itself
and pmid must be that abstract's own PMID — enforced by prompt instruction
here, and worth checking again downstream (citation-health check in
agent/loop.py) before anything is presented to a user.
"""

import logging

from pydantic import ValidationError

from causal_kg.llm.base import LLMClient
from causal_kg.models import Abstract, CausalRelation

logger = logging.getLogger(__name__)

RELATIONS_SCHEMA = {
    "title": "extracted_relations",
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "increases",
                            "decreases",
                            "causes",
                            "treats",
                            "targets",
                            "associated_with",
                        ],
                    },
                    "object": {"type": "string"},
                    "object_type": {"type": "string"},
                    "evidence_sentence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "subject",
                    "subject_type",
                    "relation",
                    "object",
                    "object_type",
                    "evidence_sentence",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}


def _build_prompt(abstract: Abstract) -> str:
    return f"""Extract directional cause-and-effect relationships from this biomedical abstract.

Title: {abstract.title}
Abstract: {abstract.text}

Rules:
- Only extract relationships explicitly stated or strongly implied in the text above.
- evidence_sentence must be an actual sentence (or close paraphrase) from the abstract text.
- subject_type/object_type should be short labels like "drug", "protein", "gene", "disease", "outcome", "pathway".
- confidence in [0,1]: how directly the text supports this relationship.
- Extract at most 5 relationships. If none are clearly supported, return an empty list.
"""


def extract_relations(
    llm: LLMClient, abstract: Abstract, max_retries: int = 2
) -> list[CausalRelation]:
    prompt = _build_prompt(abstract)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            raw = llm.complete_structured(prompt, RELATIONS_SCHEMA)
            relations = []
            for item in raw.get("relations", []):
                relation = CausalRelation(
                    **item,
                    pmid=abstract.pmid,
                )
                relations.append(relation)
            return relations
        except (ValidationError, KeyError, TypeError) as exc:
            last_error = exc
            prompt += f"\n\nYour previous output failed validation: {exc}. Return valid JSON matching the schema."
            logger.warning("extraction retry %d for PMID %s: %s", attempt, abstract.pmid, exc)

    logger.error("extraction failed for PMID %s after retries: %s", abstract.pmid, last_error)
    return []
