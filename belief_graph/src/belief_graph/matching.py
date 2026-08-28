from enum import Enum
from typing import List, Optional, Protocol

from pydantic import BaseModel

from .core import ExtractedTriplet
from .memory import BeliefMatch


class MatchType(str, Enum):
    SAME = "same"
    DIFFERENT = "different"
    CONTRADICTS = "contradicts"


class MatchResult(BaseModel):
    """
    Semantic relationship between a newly extracted belief and
    an existing belief retrieved from memory.
    """

    decision: MatchType
    matched_belief_id: Optional[str] = None
    reason: str
    similarity_score: Optional[float] = None


class BeliefMatcherProvider(Protocol):
    """
    Contract for semantic belief comparison.

    The provider receives a newly extracted belief together with
    semantically similar ACTIVE belief candidates and decides whether
    the new belief is equivalent, contradictory, or distinct.
    """

    def match_beliefs(
        self,
        new_belief: ExtractedTriplet,
        candidates: List[BeliefMatch],
    ) -> MatchResult:
        ...


class SemanticBeliefMatcher:
    """
    Coordinates semantic belief matching.

    Vector similarity is used only for candidate retrieval.
    The matcher provider makes the final semantic decision.
    """

    def __init__(self, provider: BeliefMatcherProvider):
        self.provider = provider

    def match(
        self,
        new_belief: ExtractedTriplet,
        candidates: List[BeliefMatch],
    ) -> MatchResult:

        # No semantically similar existing beliefs were retrieved.
        if not candidates:
            return MatchResult(
                decision=MatchType.DIFFERENT,
                matched_belief_id=None,
                reason="No candidate beliefs were retrieved from memory.",
            )

        result = self.provider.match_beliefs(
            new_belief=new_belief,
            candidates=candidates,
        )

        candidate_by_id = {
            match.belief.belief_id: match
            for match in candidates
        }

        # SAME and CONTRADICTS must refer to an actual retrieved candidate.
        if result.decision in {
            MatchType.SAME,
            MatchType.CONTRADICTS,
        }:
            if not result.matched_belief_id:
                raise ValueError(
                    f"{result.decision.value} decision requires "
                    "matched_belief_id."
                )

            if result.matched_belief_id not in candidate_by_id:
                raise ValueError(
                    "Matcher returned belief_id that was not among "
                    f"retrieved candidates: {result.matched_belief_id}"
                )

            result.similarity_score = (
                candidate_by_id[result.matched_belief_id].score
            )

        # DIFFERENT means there is no identity/contradiction target.
        else:
            result.matched_belief_id = None
            result.similarity_score = None

        return result
