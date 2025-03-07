from typing import Dict, List, Optional, Union, TYPE_CHECKING
import asyncio
import random

from inner_thoughts_ai.models.enums import EventType, ParticipantType, MentalObjectType
from inner_thoughts_ai.models.memory import Memory, MemoryStore
from inner_thoughts_ai.models.thought import Thought, ThoughtReservoir
from inner_thoughts_ai.utils.thinking_engine import (
    generate_system1_thought,
    generate_system2_thoughts,
    evaluate_thought,
    articulate_thought
)
from inner_thoughts_ai.utils.turn_taking_engine import predict_next_speaker
from inner_thoughts_ai.utils.saliency import recalibrate_all_saliency

# Use TYPE_CHECKING for Conversation to avoid circular imports
if TYPE_CHECKING:
    from inner_thoughts_ai.models.conversation import Conversation, Event

class Participant:
    # Class variable to keep track of the next available ID
    _next_id = 0
    
    def __init__(
        self,
        name: str,
        type: ParticipantType,
        id: Optional[str] = None
    ):
        # If ID is not provided, use the next available ID
        if id is None:
            self.id = str(Participant._next_id)
            Participant._next_id += 1
        else:
            self.id = id
            
        self.name = name
        self.type = type
        self.last_spoken_turn = -1
    
    def send_message(self, message: str, conversation: 'Conversation') -> None:
        """Send a message to the conversation."""
        conversation.record_event(
            Event(
                participant_id=self.id,
                type=EventType.UTTERANCE,
                content=message,
                turn_number=conversation.turn_number,
                participant_name=self.name
            )
        )
        
        # Update the last spoken turn
        self.last_spoken_turn = conversation.turn_number

    async def act(self, conversation: 'Conversation') -> List[Thought]:
        """Default implementation returns empty list. Override in subclasses."""
        return []

class Human(Participant):
    def __init__(self, name: str, id: Optional[str] = None, **kwargs):
        super().__init__(name=name, type=ParticipantType.HUMAN, id=id, **kwargs)

class Agent(Participant):
    def __init__(
        self,
        name: str,
        id: Optional[str] = None,
        proactivity_config: Dict = {},
        **kwargs
    ):
        super().__init__(name, ParticipantType.AGENT, id)
        self.thought_reservoir = ThoughtReservoir()
        self.memory_store = MemoryStore()
        self.proactivity_config = proactivity_config
        
    async def select_thoughts(self, thoughts: List[Thought], conversation: 'Conversation') -> List[Thought]:
        """Select thoughts that could be potentially articulated.
        
        Args:
            thoughts: List of thoughts to choose from
            conversation: Current conversation
            
        Returns:
            List of selected thoughts
        """
        # Evaluate all thoughts concurrently
        evaluation_coroutines = [
            evaluate_thought(thought, conversation, self)
            for thought in thoughts
        ]
        await asyncio.gather(*evaluation_coroutines)
        
        # Filter thoughts with articulation probability > 0 and sufficient intrinsic motivation
        selected_thoughts = [
            thought for thought in thoughts
            if thought.articulation_probability > 0 and 
            thought.intrinsic_motivation.get('score', 0) > 0.5
        ]
        
        return selected_thoughts
    
    async def act(self, conversation: 'Conversation') -> List[Thought]:
        """Select from the latest batch of thoughts and return the ones that could be articulated.
        
        Args:
            conversation: Current conversation context
            
        Returns:
            List of thoughts that could be articulated
        """
        # Get the latest batch of thoughts with turn_number
        new_thoughts = [t for t in self.thought_reservoir.thoughts if t.turn_number == conversation.turn_number]
        
        if not new_thoughts:
            return []
        
        # Select thoughts that could be potentially articulated
        selected_thoughts = await self.select_thoughts(new_thoughts, conversation)
        
        return selected_thoughts

    async def think(self, conversation: Conversation) -> None:
        """Think about the conversation, generate thoughts, and evaluate them."""
        # Get the last event from the conversation
        last_events = conversation.get_last_n_events(1)
        if not last_events:
            return []  # No events to process
        
        last_event = last_events[0]
        
        # 1. Process the event - Skip if this agent is the source of the event
        if last_event.participant_id == self.id:
            return []  # Don't respond to our own events
            
        # 2. Recalibrate saliency scores of long-term memories and thoughts based on the new event
        # First need to ensure the event has an embedding
        if last_event.embedding is None:
            await last_event.compute_embedding_async()
            
        # Recalibrate long-term memories
        recalibrate_all_saliency(
            items=self.memory_store.long_term_memory,
            utterance=last_event
        )
        
        # Recalibrate thoughts
        recalibrate_all_saliency(
            items=self.thought_reservoir.thoughts,
            utterance=last_event
        )
        
        # 3. Add event to short-term memory
        # Create a memory from the event
        memory = Memory(
            agent_id=int(self.id),
            type=MentalObjectType.MEMORY_SHORT_TERM,
            content=f"{last_event.participant_name} said: {last_event.content}",
            turn_number=last_event.turn_number,
            last_accessed_turn=conversation.turn_number,
            compute_embedding=True
        )
        
        # Add to memory store
        self.memory_store.add(memory)
        
        # 4. Generate thoughts
        new_thoughts = await self.generate_thoughts(conversation)
        
        # 5. Evaluate thoughts
        await self.evaluate_thoughts(new_thoughts, conversation)

        # 6. Add thoughts to reservoir
        for thought in new_thoughts:
            self.thought_reservoir.add(thought)
        

    async def generate_thoughts(self, conversation: Conversation, num_system1: int = 1, num_system2: int = 2) -> List[Thought]:
        """Generate thoughts of both system 1 and system 2 types.
        
        Args:
            conversation: The conversation context
            num_system1: Number of system 1 thoughts to generate (currently fixed at 1)
            num_system2: Number of system 2 thoughts to generate
            
        Returns:
            List of generated thoughts
        """
        # Generate System 1 thought (quick, automatic response)
        system1_thought = await generate_system1_thought(
            conversation=conversation,
            agent=self
        )
        
        # Generate System 2 thoughts (deliberate, memory-based responses)
        system2_thoughts = await generate_system2_thoughts(
            conversation=conversation,
            agent=self,
            num_thoughts=num_system2
        )
            
        # Return all generated thoughts
        return [system1_thought] + system2_thoughts
    
    async def evaluate_thoughts(self, new_thoughts: List[Thought], conversation: Conversation) -> None:
        """Evaluate newly generated thoughts based on various criteria.
        
        Args:
            new_thoughts: List of new thoughts to evaluate
            conversation: The conversation context
        """ 
        for thought in new_thoughts:
            await evaluate_thought(
                thought=thought,
                conversation=conversation,
                agent=self
            )
            
    
    async def articulate_thought(self, thought: Thought, conversation: Conversation) -> str:
        """Articulate a thought into an utterance string.
        
        Args:
            thought: The thought to articulate
            conversation: The conversation context
            
        Returns:
            Articulated text ready for expression in the conversation
        """
        return await articulate_thought(thought, conversation, agent=self)
    

    async def select_thoughts(self, thoughts: List[Thought], conversation: Conversation) -> List[Thought]:
        """Select thoughts that could be potentially articulated based on the proactivity configuration, given a list of thoughts.
        
        This method implements the Iterative Thought Reservoir Decision Process algorithm,
        which selects thoughts for articulation based on turn-taking predictions,
        intrinsic motivation scores, and proactivity configuration.
        
        Args:
            thoughts: List of thoughts to select from
            conversation: The current conversation context
            
        Returns:
            List of selected thoughts to potentially articulate
        """
        # Get proactivity configuration thresholds
        im_threshold = self.proactivity_config.get('im_threshold', 0.7)  # Default threshold for intrinsic motivation
        system1_prob = self.proactivity_config.get('system1_prob', 0.3)  # Default probability for system1 thoughts
        interrupt_threshold = self.proactivity_config.get('interrupt_threshold', 0.85)  # Default threshold for interruption

        if len(thoughts) == 0:
            return []

        # Filter thoughts that have been evaluated (have intrinsic_motivation scores)
        evaluated_thoughts = [t for t in thoughts if isinstance(t.intrinsic_motivation, dict) and 'score' in t.intrinsic_motivation]
        if not evaluated_thoughts:
            return []  # No evaluated thoughts to select from
            
        # Sort thoughts by their intrinsic motivation score
        sorted_thoughts = sorted(
            evaluated_thoughts, 
            key=lambda t: t.intrinsic_motivation['score'], 
            reverse=True
        )
        
        # Step 1: Predict the turn-taking type for the current event
        predicted_speaker = conversation.event_history[-1].pred_next_turn
        
        selected_thoughts = []
        
        # Step 2: Process according to turn-taking type
        if predicted_speaker == "anyone":
            # Turn is open to anyone
            high_motivation_thoughts = [t for t in sorted_thoughts if t.intrinsic_motivation['score'] >= im_threshold]
            
            if high_motivation_thoughts:
                # Select the highest-rated thought
                selected_thoughts.append(high_motivation_thoughts[0])
            else:
                # With some probability, select from system-1 thoughts
                if random.random() < system1_prob:
                    system1_thoughts = [t for t in sorted_thoughts if t.type == MentalObjectType.THOUGHT_SYSTEM1]
                    if system1_thoughts:
                        selected_thoughts.append(system1_thoughts[0])
                        
        elif predicted_speaker == self.name:
            # Turn is allocated to this agent
            # Select the highest-rated thought
            if sorted_thoughts:
                selected_thoughts.append(sorted_thoughts[0])
        else:
            # Turn is allocated to someone else
            interrupt_thoughts = [t for t in sorted_thoughts if t.intrinsic_motivation['score'] >= interrupt_threshold]
            
            if interrupt_thoughts:
                # Select the highest-rated thought for interruption
                selected_thoughts.append(interrupt_thoughts[0])
                
        return selected_thoughts
        
