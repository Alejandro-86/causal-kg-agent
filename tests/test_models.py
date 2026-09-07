import pytest
from pydantic import ValidationError

from causal_kg.models import CausalRelation, RelationType


def _valid_kwargs(**overrides):
    kwargs = dict(
        subject="semaglutide",
        subject_type="drug",
        relation=RelationType.DECREASES,
        object="gastric emptying rate",
        object_type="outcome",
        evidence_sentence="Semaglutide slows gastric emptying.",
        pmid="12345678",
        confidence=0.9,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_relation_constructs():
    relation = CausalRelation(**_valid_kwargs())
    assert relation.relation == RelationType.DECREASES
    assert relation.pmid == "12345678"


def test_confidence_above_one_rejected():
    with pytest.raises(ValidationError):
        CausalRelation(**_valid_kwargs(confidence=1.5))


def test_confidence_below_zero_rejected():
    with pytest.raises(ValidationError):
        CausalRelation(**_valid_kwargs(confidence=-0.1))


def test_invalid_relation_enum_rejected():
    with pytest.raises(ValidationError):
        CausalRelation(**_valid_kwargs(relation="cures"))


def test_relation_enum_from_string_value():
    relation = CausalRelation(**_valid_kwargs(relation="treats"))
    assert relation.relation == RelationType.TREATS
