from typing import List, Optional
import numpy as np  # type: ignore
from numpy.typing import NDArray

from inner_thoughts_ai.models.enums import EventType
from inner_thoughts_ai.models.participant import Participant

class Event:
    def __init__(
        self,
        participant_id: str,
        type: EventType,
        content: str,
        embedding: NDArray[np.float32],
        interpretation: str,
        interpretation_embedding: NDArray[np.float32],
        turn_number: int,
        thought_id: Optional[int] = None,
        pred_next_turn: str = ""
    ):
        self.participant_id = participant_id
        self.type = type
        self.content = content
        self.embedding = embedding
        self.interpretation = interpretation
        self.interpretation_embedding = interpretation_embedding
        self.turn_number = turn_number
        self.thought_id = thought_id
        self.pred_next_turn = pred_next_turn
    
    def interpret(self) -> str:
        """Interpret the event and return the interpretation."""
        pass

class Conversation:
    def __init__(self, context: str):
        self.context = context
        self.participants: List['Participant'] = []
        self.event_history: List[Event] = []
    
    def add_participant(self, participant: 'Participant') -> None:
        """Add a participant to the conversation."""
        self.participants.append(participant)
    
    def remove_participant(self, participant: 'Participant') -> None:
        """Remove a participant from the conversation."""
        self.participants.remove(participant)
    
    def record_event(self, event: Event) -> None:
        """Record an event in the conversation history."""
        self.event_history.append(event)
    
    def broadcast_event(self, event: Event) -> None:
        """Broadcast an event to all participants."""
        for participant in self.participants:
            participant.on_receive_event(event)
    
    # Getters
    def get_event_history(self) -> List[Event]:
        """Get the event history."""
        return self.event_history
    
    def get_last_n_events(self, n: int = 5) -> List[Event]:
        """Get the last n events."""
        return self.event_history[-n:]
    
    def get_participants(self) -> List['Participant']:
        """Get the participants."""
        return self.participants
    
    def get_context(self) -> str:
        """Get the context."""
        return self.context
    
    
    
