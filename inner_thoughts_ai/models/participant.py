from typing import Dict, List, Optional, Union, TYPE_CHECKING, Callable, Any
import random
import asyncio
import uuid

from inner_thoughts_ai.models.enums import EventType, ParticipantType, MentalObjectType
from inner_thoughts_ai.models.memory import Memory, MemoryStore
from inner_thoughts_ai.models.thought import Thought, ThoughtReservoir
from inner_thoughts_ai.models.conversation import Event, Conversation
from inner_thoughts_ai.utils.thinking_engine import (
    generate_system1_thought,
    generate_system2_thoughts,
    evaluate_thought,
    articulate_thought
)
from inner_thoughts_ai.utils.saliency import recalibrate_all_saliency
from inner_thoughts_ai.utils.text_splitter import SentenceSplitter




class Participant:
    
    def __init__(
        self,
        name: str,
        type: ParticipantType,
        id: Optional[str] = None,
        **kwargs
    ):
        # Use provided ID or generate a UUID
        self.id = id if id is not None else str(uuid.uuid4())
            
        self.name = name
        self.type = type
        self.last_spoken_turn = -1
    
    async def send_message(self, message: str, conversation: 'Conversation', interpret: bool = True) -> Event:
        """Send a message to the conversation.
        
        This method creates an utterance event, records it in the conversation,
        and immediately interprets it to ensure the interpretation is available
        for all subsequent operations.
        
        Args:
            message: The message content
            conversation: The conversation to send the message to
            interpret: Whether to interpret the message immediately
        """
        
        # Create the event
        event = Event(
            participant_id=self.id,
            type=EventType.UTTERANCE,
            content=message,
            turn_number=conversation.turn_number,
            participant_name=self.name
        )

        # Record the event
        conversation.record_event(event)

        if interpret:
            # Interpret the event
            await conversation.interpret_event(event)
        
        # Update the last spoken turn
        self.last_spoken_turn = conversation.turn_number

        return event


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
        super().__init__(name=name, type=ParticipantType.AGENT)
        self.thought_reservoir = ThoughtReservoir()
        self.memory_store = MemoryStore()
        self.proactivity_config = proactivity_config
        self.text_splitter = SentenceSplitter()
       

    async def think(self, conversation: Conversation, event: 'Event') -> None:
        """Process an event, including generating thoughts, evaluating them, and adding them to the thought reservoir.

        
        Args:
            conversation: The conversation context
            event: The event to process
        """
        # 1. Process the event - Skip if this agent is the source of the event
        if event.participant_id == self.id:
            return []  # Don't respond to our own events
            
        # 2. Recalibrate saliency scores based on the new event
        await self.recalibrate_saliency_for_event(event)
        
        # 3. Add event to short-term memory
        self.add_event_to_memory(event)
        
        # 4. Generate thoughts
        new_thoughts = await self.generate_thoughts(conversation)
        
        # 5. Evaluate thoughts
        await self.evaluate_thoughts(new_thoughts, conversation)

        # 6. Add thoughts to reservoir
        for thought in new_thoughts:
            self.thought_reservoir.add(thought)
    
        # 7. Select thoughts to articulate
        await self.select_thoughts(new_thoughts, conversation)


    async def recalibrate_saliency_for_event(self, event: 'Event') -> None:
        """Recalibrate saliency scores of long-term memories and thoughts based on a new event.
        
        Args:
            event: The event to use for recalibration
        """
        # Ensure the event has an embedding
        if event.embedding is None:
            await event.compute_embedding_async()
            
        # Recalibrate long-term memories
        recalibrate_all_saliency(
            items=self.memory_store.long_term_memory,
            utterance=event
        )
        
        # Recalibrate thoughts
        recalibrate_all_saliency(
            items=self.thought_reservoir.thoughts,
            utterance=event
        )

    async def generate_thoughts(self, conversation: Conversation, num_system1: int = 1, num_system2: int = 2) -> List[Thought]:
        """Generate thoughts of both system 1 and system 2 types.
        
        Args:
            conversation: The conversation
            num_system1: Number of system 1 thoughts to generate
            num_system2: Number of system 2 thoughts to generate 
            
        Returns:
            List of generated thoughts
        """
        # Generate System 1 and System 2 thoughts concurrently
        system1_task = generate_system1_thought(
            conversation=conversation,
            agent=self
        )
        
        system2_task = generate_system2_thoughts(
            conversation=conversation,
            agent=self,
            num_thoughts=num_system2
        )
        
        # Wait for both tasks to complete
        system1_thought, system2_thoughts = await asyncio.gather(system1_task, system2_task)
            
        # Return all generated thoughts
        return [system1_thought] + system2_thoughts
    
    async def evaluate_thoughts(self, new_thoughts: List[Thought], conversation: Conversation) -> None:
        """Evaluate newly generated thoughts based on various criteria.
        
        Args:
            new_thoughts: List of new thoughts to evaluate
            conversation: The conversation
        """ 
        # Create evaluation tasks for all thoughts
        evaluation_tasks = [
            evaluate_thought(
                thought=thought,
                conversation=conversation,
                agent=self
            )
            for thought in new_thoughts
        ]
        
        # Execute all evaluation tasks concurrently
        if evaluation_tasks:
            await asyncio.gather(*evaluation_tasks)
            
    
    async def articulate_thought(self, thought: Thought, conversation: Conversation) -> str:
        """Articulate a thought into an utterance string.
        
        Args:
            thought: The thought to articulate
            conversation: The conversation
            
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
            conversation: The current conversation
            
        Returns:
            List of selected thoughts to potentially articulate
        """
        # Reset selected status for all thoughts
        for thought in thoughts:
            thought.selected = False
            
        # Get proactivity configuration thresholds
        im_threshold = self.proactivity_config.get('im_threshold', 3.5)  # Default threshold for intrinsic motivation
        system1_prob = self.proactivity_config.get('system1_prob', 0.3)  # Default probability for system1 thoughts
        interrupt_threshold = self.proactivity_config.get('interrupt_threshold', 4)  # Default threshold for interruption

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

        # Mark the selected thoughts as selected
        for thought in selected_thoughts:
            thought.selected = True
                
        return selected_thoughts
    

    def add_event_to_memory(self, event: 'Event', memory_type: MentalObjectType = MentalObjectType.MEMORY_SHORT_TERM, compute_embedding: bool = True) -> None:
        """Add an event to the agent's memory.
        
        Args:
            event: The event to add to memory
            memory_type: The type of memory to create (default: MEMORY_SHORT_TERM)
            
        Returns:
            None
        """
        # Create a memory from the event
        memory = Memory(
            agent_id=self.id,
            type=memory_type,
            content=f"{event.participant_name}: {event.content}",
            turn_number=event.turn_number,
            last_accessed_turn=event.turn_number,
            compute_embedding=compute_embedding
        )
        
        # Add to memory store
        self.memory_store.add(memory)
        
    def initialize_memory(self, text: str, memory_type: MentalObjectType = MentalObjectType.MEMORY_LONG_TERM, by_paragraphs: bool = False, compute_embedding: bool = True) -> None:
        """Initialize the agent's memory with a text.
        
        This method splits the input text into chunks (sentences or paragraphs),
        creates a memory for each chunk of the specified type, and adds it to the agent's memory store.

        Args:
            text: The text to parse and add as memory
            memory_type: The type of memory to create (default: MEMORY_LONG_TERM)
            by_paragraphs: Whether to split by paragraphs first (default: False)
            compute_embedding: Whether to compute embeddings for the memories (default: True)

        Returns:
            None
        """
        # Clean whitespace from the text
        text = text.strip()

        # Split the text into chunks
        chunks = self.text_splitter.split_text(text, by_paragraphs=by_paragraphs)


        for chunk in chunks:
            # Create a memory for each chunk
            memory = Memory(
                agent_id=self.id,
                type=memory_type,
                content=chunk,
                turn_number=0,
                last_accessed_turn=0,
                compute_embedding=compute_embedding
            )
            self.memory_store.add(memory)