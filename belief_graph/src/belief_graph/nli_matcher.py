import logging
from typing import List
from pydantic import BaseModel

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .core import ExtractedTriplet, Belief
from .memory import BeliefMatch
from .matching import MatchResult, MatchType

logger = logging.getLogger(__name__)


class NliScores(BaseModel):
    label: str
    entailment: float
    contradiction: float
    neutral: float


class NliComparison(BaseModel):
    new_to_existing: NliScores
    existing_to_new: NliScores

    @property
    def same_score(self) -> float:
        return min(
            self.new_to_existing.entailment,
            self.existing_to_new.entailment,
        )


class NliMatcherProvider:
    """
    NLI-based semantic belief matcher.

    Maps Natural Language Inference relations to BeliefTrace relations:

    bidirectional entailment -> SAME
    contradiction           -> CONTRADICTS
    otherwise               -> DIFFERENT
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _belief_to_nli_text(belief: ExtractedTriplet | Belief) -> str:
        subject = belief.subject
        object_ = belief.object
        relation = belief.relation
        subject_text = {
            "Self": "I",
            "Others": "Other people",
            "World": "The world",
        }[subject]

        obj = object_.strip().lower()
        rel = relation.strip().upper()

        if subject == "Self":
            templates = {
                "IS": f"I am {obj}.",
                "IS NOT": f"I am not {obj}.",
                "WAS": f"I was {obj}.",
                "CAN": f"I can {obj}.",
                "CANNOT": f"I cannot {obj}.",
                "WILL": f"I will {obj}.",
                "WILL NOT": f"I will not {obj}.",
                "WANTS": f"I want to {obj}.",
                "DESIRES": f"I desire to {obj}.",
                "AVOIDS": f"I avoid {obj}.",
                "FEELS": f"I feel {obj}.",
                "QUESTIONING": f"I am questioning {obj}.",
                "ASSESSES": f"I assess {obj}.",
                "PERCEIVES": f"I perceive {obj}.",
            }

        elif subject == "World":
            templates = {
                "IS": f"The world is {obj}.",
                "IS NOT": f"The world is not {obj}.",
                "HAS": f"The world has {obj}.",
                "CAUSES": f"The world causes {obj}.",
            }

        else:  # Others
            templates = {
                "ARE": f"Other people are {obj}.",
                "ARE NOT": f"Other people are not {obj}.",
                "CAN": f"Other people can {obj}.",
                "WILL": f"Other people will {obj}.",
                "CARE": f"Other people care {obj}.",
                "WISHES": f"Other people wish to {obj}.",
            }

        return templates.get(
            rel,
            f"{subject_text} {rel.lower()} {obj}."
        )

    def _predict_nli(
            self,
            premise: str,
            hypothesis: str,
    ) -> NliScores:

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]

        scores = {}

        for idx, probability in enumerate(probabilities):
            label = self.model.config.id2label[idx].lower()
            scores[label] = float(probability)

        predicted_id = int(torch.argmax(probabilities))
        predicted_label = (
            self.model.config.id2label[predicted_id].lower()
        )

        return NliScores(
            label=predicted_label,
            entailment=scores["entailment"],
            contradiction=scores["contradiction"],
            neutral=scores["neutral"],
        )

    def compare(
            self,
            new_belief: ExtractedTriplet,
            candidate: Belief,
    ) -> NliComparison:

        new_text = self._belief_to_nli_text(new_belief)
        candidate_text = self._belief_to_nli_text(candidate)

        logger.debug(f"[NLI] NEW:      {new_text}")
        logger.debug(f"[NLI] EXISTING: {candidate_text}")

        return NliComparison(
            new_to_existing=self._predict_nli(
                premise=new_text,
                hypothesis=candidate_text,
            ),
            existing_to_new=self._predict_nli(
                premise=candidate_text,
                hypothesis=new_text,
            ),
        )

    def _compare(
        self,
        new_belief: ExtractedTriplet,
        candidate: Belief,
    ) -> tuple[MatchType, str]:

        comparison = self.compare(
            new_belief=new_belief,
            candidate=candidate,
        )

        new_to_existing = comparison.new_to_existing
        existing_to_new = comparison.existing_to_new

        # SAME requires approximate bidirectional entailment.
        if (
            new_to_existing.label == "entailment"
            and existing_to_new.label == "entailment"
        ):
            return (
                MatchType.SAME,
                (
                    "Bidirectional entailment: "
                    f"new→existing={new_to_existing.entailment:.3f}, "
                    f"existing→new={existing_to_new.entailment:.3f}."
                ),
            )

        # Contradiction is treated as semantic incompatibility.
        if (
                new_to_existing.label == "contradiction"
                or existing_to_new.label == "contradiction"
        ):
            return (
                MatchType.CONTRADICTS,
                (
                    "NLI detected semantic contradiction: "
                    f"new→existing={new_to_existing.contradiction:.3f}, "
                    f"existing→new={existing_to_new.contradiction:.3f}."
                ),
            )

        return (
            MatchType.DIFFERENT,
            (
                "The beliefs are neither bidirectionally entailed "
                "nor semantically contradictory: "
                f"new→existing={new_to_existing.label}, "
                f"existing→new={existing_to_new.label}."
            ),
        )

    def match_beliefs(
        self,
        new_belief: ExtractedTriplet,
        candidates: List[BeliefMatch],
    ) -> MatchResult:

        if not candidates:
            return MatchResult(
                decision=MatchType.DIFFERENT,
                matched_belief_id=None,
                reason="No candidate beliefs were retrieved from memory.",
            )

        # v0.2 first pass: evaluate candidates in Qdrant similarity order.
        contradictions = []

        for candidate_match in candidates:
            candidate = candidate_match.belief

            decision, reason = self._compare(
                new_belief=new_belief,
                candidate=candidate,
            )

            # Identity has highest priority.
            if decision == MatchType.SAME:
                return MatchResult(
                    decision=MatchType.SAME,
                    matched_belief_id=candidate.belief_id,
                    reason=reason,
                    similarity_score=candidate_match.score,
                )

            if decision == MatchType.CONTRADICTS:
                contradictions.append(
                    (
                        candidate_match,
                        reason,
                    )
                )

        # No SAME candidate exists.
        # For now select the highest-ranked contradictory candidate.
        if contradictions:
            candidate_match, reason = contradictions[0]

            return MatchResult(
                decision=MatchType.CONTRADICTS,
                matched_belief_id=candidate_match.belief.belief_id,
                reason=reason,
                similarity_score=candidate_match.score,
            )

        return MatchResult(
            decision=MatchType.DIFFERENT,
            matched_belief_id=None,
            reason=(
                "None of the retrieved candidates were equivalent "
                "or contradictory."
            ),
        )
