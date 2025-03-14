from typing import List, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import uuid
import asyncio

from thoughtful_agents.models.enums import EventType
from thoughtful_agents.utils.llm_api import get_completion, get_embedding_sync, get_embedding_async

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from thoughtful_agents.models.participant import Participant

class Event:
    
    def __init__(
        self,
        participant_id: str,
        type: EventType,
        content: str,
        turn_number: int,
        participant_name: str = "Unknown",
        interpretation: str = "",
        thought_id: Optional[int] = None,
        pred_next_turn: str = "",
        embedding: Optional[Union[NDArray[np.float32], List[float]]] = None,
        interpretation_embedding: Optional[Union[NDArray[np.float32], List[float]]] = None,
        compute_embedding: bool = True
    ):
        # Always generate a UUID
        self.id = str(uuid.uuid4())
            
        # Simple attributes that don't need validation
        self.participant_id = participant_id
        self.type = type
        self.content = content
        self.turn_number = turn_number
        self.participant_name = participant_name
        self.thought_id = thought_id
        self.interpretation = interpretation
        self.pred_next_turn = pred_next_turn
        
        # Handle content embedding computation
        if embedding is not None:
            # Convert list to numpy array if needed
            if isinstance(embedding, list):
                self.embedding = np.array(embedding, dtype=np.float32)
            else:
                self.embedding = embedding
        elif compute_embedding:
            # Compute embedding synchronously
            self.embedding = self._compute_embedding_sync(content)
        else:
            # Defer embedding computation
            self.embedding = None
            
        # Handle interpretation embedding computation
        if interpretation_embedding is not None:
            # Convert list to numpy array if needed
            if isinstance(interpretation_embedding, list):
                self.interpretation_embedding = np.array(interpretation_embedding, dtype=np.float32)
            else:
                self.interpretation_embedding = interpretation_embedding
        elif compute_embedding and interpretation:
            # Compute embedding synchronously
            self.interpretation_embedding = self._compute_embedding_sync(interpretation)
        else:
            # Defer embedding computation
            self.interpretation_embedding = None
    
    def _compute_embedding_sync(self, text: str) -> NDArray[np.float32]:
        """Compute embedding synchronously (blocking call).
        
        This is a helper method for the constructor. For async code,
        use compute_embedding_async instead.
        """
        # Use the synchronous version of get_embedding
        embedding_list = get_embedding_sync(text)
        return np.array(embedding_list, dtype=np.float32)
    
    async def compute_embedding_async(self) -> None:
        """Compute embedding asynchronously if it wasn't computed in the constructor."""
        if self.embedding is None:
            embedding_list = await get_embedding_async(self.content)
            self.embedding = np.array(embedding_list, dtype=np.float32)
    
    async def compute_interpretation_embedding_async(self) -> None:
        """Compute interpretation embedding asynchronously if it wasn't computed in the constructor."""
        if self.interpretation and self.interpretation_embedding is None:
            embedding_list = await get_embedding_async(self.interpretation)
            self.interpretation_embedding = np.array(embedding_list, dtype=np.float32)
    
    def has_interpretation(self) -> bool:
        """Check if this event has an interpretation."""
        return bool(self.interpretation)

class Conversation:
    def __init__(self, context: str):
        self.context = context
        self.participants: List['Participant'] = []
        self.event_history: List[Event] = []
        self.turn_number = 0

    def add_participant(self, participant: 'Participant') -> None:
        """Add a participant to the conversation."""
        self.participants.append(participant)
    
    def remove_participant(self, participant: 'Participant') -> None:
        """Remove a participant from the conversation."""
        self.participants.remove(participant)
    

    def record_event(self, event: Event) -> None:
        """Record an event in the conversation history."""
        # Set the participant_name if it's still the default "Unknown"
        if event.participant_name == "Unknown":
            for participant in self.participants:
                if participant.id == event.participant_id:
                    event.participant_name = participant.name
                    break
        
        self.event_history.append(event)
        self.turn_number += 1

    
    async def interpret_event(self, event: Event) -> str:
        """Generate an interpretation for an event.
        
        Args:
            event: The event to interpret
            
        Returns:
            The interpretation text
        """
        # Skip interpretation for non-utterance events
        if event.type != EventType.UTTERANCE:
            return ""
            
        # Retrieve recent conversation history
        last_events = self.get_last_n_events(5)
        conversation_history = ""
        for e in last_events:
            if e.id == event.id:
                break  # Stop when we reach the current event
            conversation_history += f"{e.participant_name}: {e.content}\n"
            
        # Create the prompt
        system_prompt = "You are an assistant that interprets the meaning and intent behind utterances in a conversation. Provide a brief interpretation that captures the key points, emotional tone, and implicit meaning."
        user_prompt = f"Conversation history:\n{conversation_history}\n\nUtterance to interpret:\n{event.participant_name}: {event.content}\n\nInterpretation:"
        
        # Call the OpenAI API
        try:
            response = await get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )
            
            # Extract the interpretation from the response
            interpretation = response.get("text", "").strip()
            
            # Update the event with the interpretation
            event.interpretation = interpretation
            
            # Compute the interpretation embedding asynchronously
            await self.compute_interpretation_embedding(event)
            
            return interpretation
            
        except Exception as e:
            # Log the error and return an empty interpretation
            print(f"Error interpreting event: {str(e)}")
            return ""
            
    async def compute_interpretation_embedding(self, event: Event) -> None:
        """Compute the embedding for an event's interpretation.
        
        Args:
            event: The event to compute the interpretation embedding for
        """
        if event.interpretation and event.interpretation_embedding is None:
            try:
                embedding_list = await get_embedding_async(event.interpretation)
                event.interpretation_embedding = np.array(embedding_list, dtype=np.float32)
            except Exception as e:
                # Log the error
                print(f"Error computing interpretation embedding: {str(e)}")
    
    
    async def broadcast_event(self, event: Event) -> None:
        """Broadcast the specified event to all participants and let them process it concurrently.
        
        This method uses asyncio.gather to process all participants in parallel,
        which can significantly improve performance when there are multiple participants.
        
        Only Agent type participants will 'think' about the event, as other participant types
        (like Human) don't have the think method.
        
        Before broadcasting, this method ensures pred_next_turn is set on the event
        by calling predict_turn_taking_type if necessary, which is important for proper thought selection.
        
        Args:
            event: The event to broadcast to all participants
        """
        # Ensure pred_next_turn is set before broadcasting
        if not event.pred_next_turn:
            from thoughtful_agents.utils.turn_taking_engine import predict_turn_taking_type
            turn_allocation_type = await predict_turn_taking_type(self)
            # No need to set event.pred_next_turn as predict_turn_taking_type already does it
        
        # Get only Agent participants
        agent_participants = self.get_agents()
        
        # Create think tasks only for Agent participants
        process_tasks = [
            participant.think(self, event)
            for participant in agent_participants
        ]
        
        # Execute all think tasks concurrently
        if process_tasks:
            await asyncio.gather(*process_tasks)

            
    # Getters
    def get_last_n_events(self, n: int = 5) -> List[Event]:
        """Get the last n events from the conversation history.
        
        This is a utility method that handles getting a slice of the event history.
        
        Args:
            n: Number of most recent events to retrieve
            
        Returns:
            List of the n most recent events
        """
        return self.event_history[-n:]
    
    def get_agents(self) -> List['Participant']:
        """Get all participants that are Agent instances.
        
        This is a utility method that filters the participants list to only include Agent instances.
        
        Returns:
            List of participants that are Agent instances
        """
        # Import Agent here to avoid circular imports
        from thoughtful_agents.models.participant import Agent
        
        return [participant for participant in self.participants if isinstance(participant, Agent)]
    
    def get_by_id(self, event_id: str) -> Optional[Event]:
        """Get an event by its ID.
        
        This method searches through all events to find one with a matching ID.
        
        Args:
            event_id: The ID of the event to find
            
        Returns:
            The event with the matching ID, or None if not found
        """
        for event in self.event_history:
            if event.id == event_id:
                return event
        return None
    
    def get_participant_by_id(self, participant_id: str) -> Optional['Participant']:
        """Get a participant by their ID.
        
        This method searches through all participants to find one with a matching ID.
        
        Args:
            participant_id: The ID of the participant to find
            
        Returns:
            The participant with the matching ID, or None if not found
        """
        for participant in self.participants:
            if participant.id == participant_id:
                return participant
        return None
    
