from typing import Dict, List, Optional

from inner_thoughts_ai.models.enums import ParticipantType
from inner_thoughts_ai.models.conversation import Event
from inner_thoughts_ai.models.memory import MemoryStore
from inner_thoughts_ai.models.thought import ThoughtReservoir, Thought

class Participant:
    def __init__(
        self,
        id: str,
        name: str,
        type: ParticipantType
    ):
        self.id = id
        self.name = name
        self.type = type
        self.last_spoken_turn = -1
    
    def send_message(self) -> List[Event]:
        """Send a message, generating events."""
        pass
    
    def on_receive_event(self, event: Event) -> None:
        """Handle receiving an event."""
        pass

class Human(Participant):
    def __init__(self, **kwargs):
        super().__init__(type=ParticipantType.HUMAN, **kwargs)

class Agent(Participant):
    def __init__(
        self,
        persona: str,
        proactivity_config: Dict,
        **kwargs
    ):
        super().__init__(type=ParticipantType.AGENT, **kwargs)
        self.persona = persona
        self.memory_store = MemoryStore()
        self.thought_reservoir = ThoughtReservoir()
        self.proactivity_config = proactivity_config
    
    def generate_thoughts(self, num_system1: int, num_system2: int) -> None:
        """Generate thoughts of both system 1 and system 2 types."""
        pass
    
    def evaluate_thoughts(self) -> None:
        """Evaluate thoughts based on various criteria."""
        pass 

    def articulate_thought(self, thought: Thought) -> str:
        """Articulate a thought into an utterance string."""
        pass
