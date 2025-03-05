import numpy as np  # type: ignore
from numpy.typing import NDArray

from inner_thoughts_ai.models.enums import MentalObjectType

class MentalObject:
    def __init__(
        self,
        id: str,
        agent_id: int,
        type: MentalObjectType,
        content: str,
        embedding: NDArray[np.float32],
        turn_number: int,
        last_accessed_turn: int,
        retrieval_count: int,
        weight: float,
        saliency: float
    ):
        self.id = id
        self.agent_id = agent_id
        self.type = type
        self.content = content
        self.embedding = embedding
        self.turn_number = turn_number
        self.last_accessed_turn = last_accessed_turn
        self.retrieval_count = retrieval_count
        self.weight = weight
        self.saliency = saliency
    
    def get_saliency(self) -> float:
        """Get the saliency value of this mental object."""
        return self.saliency
    
    def set_saliency(self, saliency: float) -> None:
        """Set the saliency value of this mental object."""
        self.saliency = saliency 