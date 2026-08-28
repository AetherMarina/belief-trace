from typing import List, Protocol
from pydantic import BaseModel
from .core import Belief, BeliefStatus


class BeliefMatch(BaseModel):
    """
    A wrapper that pairs a retrieved belief with its similarity score,
    providing the LLM arbiter with quantitative context on the semantic match strength.
    """
    belief: Belief
    score: float


class BeliefMemory(Protocol):
    """
    Strict contract for the vector memory layer.
    Ensures the LongitudinalEngine remains completely decoupled from specific
    database implementations (e.g., Qdrant, pgvector).
    """

    def add(self, belief: Belief, embedding: List[float]) -> None:
        """
        Indexes a new belief into the vector database.
        The underlying implementation is responsible for parsing the Belief object
        and extracting the necessary payload/metadata for filtering.
        """
        ...

    def update_metadata(
        self,
        belief: Belief,
    ) -> None:
        ...

    def search_similar(
            self,
            embedding: List[float],
            entity_id: str,
            subject: str,
            status: BeliefStatus = BeliefStatus.ACTIVE,
            top_k: int = 5
    ) -> List[BeliefMatch]:
        """
        Executes a filtered vector similarity search.
        Returns a narrow set of top-k candidates strictly filtered by entity and status.
        """
        ...