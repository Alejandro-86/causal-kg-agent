"""Domain models for extracted causal relationships.

Vocabulary deliberately mirrors Biorelate's own product language
("directional cause-and-effect interactions") rather than generic
subject/predicate/object naming.
"""

from enum import Enum

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    INCREASES = "increases"
    DECREASES = "decreases"
    CAUSES = "causes"
    TREATS = "treats"
    TARGETS = "targets"
    ASSOCIATED_WITH = "associated_with"


class CausalRelation(BaseModel):
    subject: str
    subject_type: str
    relation: RelationType
    object: str
    object_type: str
    evidence_sentence: str
    pmid: str
    confidence: float = Field(ge=0.0, le=1.0)


class Abstract(BaseModel):
    pmid: str
    title: str
    text: str
    journal: str | None = None
    year: int | None = None
