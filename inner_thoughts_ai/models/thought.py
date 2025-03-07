from typing import List, Union, Optional, Dict

from inner_thoughts_ai.models.conversation import Event
from inner_thoughts_ai.models.enums import MentalObjectType
from inner_thoughts_ai.models.mental_object import MentalObject

class Thought(MentalObject):
    # Class variable to keep track of the next available Thought ID
    _next_thought_id = 0
    
    def __init__(
        self,
        agent_id: int,
        type: MentalObjectType,
        content: str,
        turn_number: int,
        last_accessed_turn: int,
        intrinsic_motivation: Dict[str, Union[str, float]],
        stimuli: List[Union[MentalObject, 'Event']],
        id: Optional[str] = None,
        **kwargs
    ):
        # Generate a Thought-specific ID if not provided
        if id is None:
            id = f"{Thought._next_thought_id}"
            Thought._next_thought_id += 1
            
        super().__init__(
            id=id,
            agent_id=agent_id,
            type=type,
            content=content,
            turn_number=turn_number,
            last_accessed_turn=last_accessed_turn,
            **kwargs
        )
        self.intrinsic_motivation = intrinsic_motivation
        self.stimuli = stimuli

class ThoughtReservoir:
    def __init__(self):
        self.thoughts: List[Thought] = []
    
    def add(self, thought: Thought) -> None:
        """Add a thought to the reservoir."""
        self.thoughts.append(thought)
    
    def remove(self, thought: Thought) -> None:
        """Remove a thought from the reservoir."""
        self.thoughts.remove(thought)
    
    def retrieve_top_k(self, k: int, threshold: float = 0.3, thought_type: MentalObjectType = MentalObjectType.THOUGHT_SYSTEM2) -> List[Thought]:
        """Retrieve top k thoughts based on the saliency score, that are at least above the threshold."""
        if thought_type == MentalObjectType.THOUGHT_SYSTEM2:
            thoughts = [thought for thought in self.thoughts if thought.type == thought_type]
        elif thought_type == MentalObjectType.THOUGHT_SYSTEM1:
            thoughts = [thought for thought in self.thoughts if thought.type == thought_type]
        else:
            thoughts = self.thoughts
        thoughts = sorted(thoughts, key=lambda x: x.saliency, reverse=True)
        return [thought for thought in thoughts if thought.saliency >= threshold][:k]
    
    def get_by_id(self, thought_id: str) -> Optional[Thought]:
        """Get a thought by its ID."""
        for thought in self.thoughts:
            if thought.id == thought_id:
                return thought
        return None
    
