from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny

from .memory import BeliefMemory, BeliefMatch
from .core import Belief, BeliefStatus


class QdrantBeliefMemory(BeliefMemory):
    """
    Qdrant-backed implementation of the BeliefMemory protocol.
    Uses in-memory mode by default for rapid local development and testing.
    """

    def __init__(self, collection_name: str = "beliefs", vector_size: int = 1024):
        """
        Initializes the Qdrant client and ensures the collection exists.

        Args:
            collection_name: Name of the Qdrant collection.
            vector_size:  Embedding dimensionality. Defaults to 1024 for mxbai-embed-large.
        """
        # Using location=":memory:" ensures no server setup is required for v0.2
        self.client = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Initialize collection if running for the first time
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def add(self, belief: Belief, embedding: List[float]) -> None:
        """
        Indexes a new belief into Qdrant. Automatically converts the Pydantic
        model to a searchable payload and handles the UUID ID constraint.
        """

        if len(embedding) != self.vector_size:
            raise ValueError(
                f"Expected embedding size {self.vector_size}, "
                f"got {len(embedding)}."
            )

        # Convert Pydantic model to dictionary for the payload
        payload = belief.model_dump()

        # Ensure enums are converted to raw strings for accurate payload filtering
        payload["status"] = belief.status.value if hasattr(belief.status, 'value') else belief.status

        point = PointStruct(
            id=belief.belief_id,
            vector=embedding,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def update_metadata(self, belief: Belief) -> None:
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "status": belief.status.value,
                "last_seen_step": belief.last_seen_step,
            },
            points=[belief.belief_id],
        )

    def search_similar(
            self,
            embedding: List[float],
            entity_id: str,
            subject: str,
            status: BeliefStatus = BeliefStatus.ACTIVE,
            top_k: int = 5
    ) -> List[BeliefMatch]:
        """
        Executes a vector similarity search filtered by exact payload matches
        (entity_id, subject, and status) to prevent cross-entity,
        cross-domain, or cross-status contamination.
        """
        status_val = status.value if hasattr(status, 'value') else status

        # Filter by entity_id, subject, and status
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="entity_id",
                    match=MatchValue(value=entity_id)
                ),
                FieldCondition(
                    key="subject",
                    match=MatchAny(any=[subject, subject.upper(), subject.capitalize()])
                ),
                FieldCondition(
                    key="status",
                    match=MatchValue(value=status_val)
                )
            ]
        )

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        ).points

        matches = []
        for hit in search_results:
            belief = Belief(**hit.payload)
            matches.append(BeliefMatch(belief=belief, score=hit.score))

        return matches
