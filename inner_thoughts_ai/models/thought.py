from typing import List, Union, Optional

from inner_thoughts_ai.models.conversation import Event
from inner_thoughts_ai.models.mental_object import MentalObject

class Thought(MentalObject):
    def __init__(
        self,
        intrinsic_motivation: float,
        stimuli: List[Union[MentalObject, 'Event']],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intrinsic_motivation = intrinsic_motivation
        self.stimuli = stimuli
    
    def articulate(self) -> str:
        """Articulate the thought into a string representation."""
        pass

class ThoughtReservoir:
    def __init__(self):
        self.thoughts: List[Thought] = []
    
    def add(self, thought: Thought) -> None:
        """Add a thought to the reservoir."""
        self.thoughts.append(thought)
    
    def remove(self, thought: Thought) -> None:
        """Remove a thought from the reservoir."""
        self.thoughts.remove(thought)
    
    def retrieve_top_k(self, k: int, by: str = "saliency") -> List[Thought]:
        """Retrieve top k thoughts based on the specified attribute."""
        pass 