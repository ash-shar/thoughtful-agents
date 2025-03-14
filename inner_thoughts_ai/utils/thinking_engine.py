"""Thinking engine for generating, evaluating, and articulating thoughts."""
from typing import List, Dict, Optional, Union, Tuple, Any
import asyncio
import json
import numpy as np
from numpy.typing import NDArray
import math

from inner_thoughts_ai.models.thought import Thought
from inner_thoughts_ai.models.conversation import Conversation
from inner_thoughts_ai.models.enums import MentalObjectType
from inner_thoughts_ai.utils.llm_api import get_completion

# Forward reference for Agent type
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inner_thoughts_ai.models.participant import Agent

async def generate_system1_thought(
    conversation: Conversation,
    agent: 'Agent'
) -> Thought:
    """Generate a single System 1 thought (quick, automatic response).
    You can only generate one System 1 thought because they are assumed to be the most spontaneous and quick reaction to the conversation.
    
    Args:
        conversation: The conversation
        agent: The agent generating the thought
        
    Returns:
        A generated Thought object
    """
    overall_context = conversation.context  
    # Get conversation history
    last_events = conversation.get_last_n_events(5)
    conversation_history = ""
    for event in last_events:
        conversation_history += f"{event.participant_name}: {event.content}\n"
    
    # Create the prompt
    system_prompt = f"""You are playing a role as a participant in an online multi-party conversation. Your name in the conversation is {agent.name}.
You will generate thoughts in JSON format."""
    user_prompt = f"""
You do not know other people in the conversation before, and your goal is to have a natural conversation with them and get to know each other.
You will be simulating the process of forming a thought in parallel with the conversation. Specifically, use system 1 thinking.
System 1 thinking is characterized by quick, automatic responses rather than deep thinking or recalling memories. 
For example, backchanneling, expressing acknowledgement, expressing surprise, showing interest and attention, a spontaneous reaction to a joke, or a reflexive response to a question.
Form ONE thought that reflects a quick, intuitive reaction to the ongoing conversation. It should be succinct, less than 15 words.

Note: System 1 thoughts are typically generic and not strongly tied to personal memories or deep interests. 
They are more about immediate reactions to the conversation flow rather than expressing personal relevance.

<Context>
Overall Context: {overall_context}
Conversation History: {conversation_history}


Respond with a JSON object in the following format:
{{
  "thought": "Your generated thought here"
}}
"""
    
    # Call the LLM API
    try:
        response = await get_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,
            response_format="json_object"
        )
        
        # Parse the response
        response_text = response.get("text", "{}")
        prompt_response = json.loads(response_text)
        
        # Create a thought object
        thought = Thought(
            agent_id=agent.id,
            type=MentalObjectType.THOUGHT_SYSTEM1,
            content=prompt_response["thought"],
            generated_turn=conversation.turn_number,
            last_accessed_turn=conversation.turn_number,
            intrinsic_motivation={"reasoning": "Default motivation before evaluation", "score": -1.0},
            stimuli=last_events,
            compute_embedding=True
        )
        
        return thought
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error generating System 1 thought: {str(e)}")
        return None

async def generate_system2_thoughts(
    conversation: Conversation,
    agent: 'Agent',
    num_thoughts: int = 1
) -> List[Thought]:
    """Generate System 2 thoughts (deliberate, memory-based responses).
    
    Args:
        conversation: The conversation
        agent: The agent generating thoughts
        num_thoughts: Number of thoughts to generate
        
    Returns:
        List of generated Thought objects
    """
    overall_context = conversation.context
    # Access memory_store and thought_reservoir directly from the agent
    memory_store = agent.memory_store
    thought_reservoir = agent.thought_reservoir
    
    # Get conversation history
    last_events = conversation.get_last_n_events(5)
    conversation_history = ""
    for i, event in enumerate(last_events):
        conversation_history += f"CON#{event.id}: {event.participant_name}: {event.content}\n"
    
    # Get salient memories
    salient_memories = memory_store.retrieve_top_k(k=5, threshold=0.25, memory_type=MentalObjectType.MEMORY_LONG_TERM)
    # update last_accessed_turn of each memory
    for memory in salient_memories:
        memory.last_accessed_turn = conversation.turn_number
    memories_text = ""
    for i, memory in enumerate(salient_memories):
        memories_text += f"MEM#{memory.id}: {memory.content}\n"
    
    # Get previous thoughts
    previous_thoughts = thought_reservoir.retrieve_top_k(k=3, threshold=0.25, thought_type=MentalObjectType.THOUGHT_SYSTEM2)
    # update last_accessed_turn of each thought
    for thought in previous_thoughts:
        thought.last_accessed_turn = conversation.turn_number
    thoughts_text = ""
    for i, thought in enumerate(previous_thoughts):
        thoughts_text += f"THO#{thought.id}: {thought.content}\n"
    
    # Create the prompt
    system_prompt = f"""You are playing a role as a participant in an online multi-party conversation. Your name in the conversation is {agent.name}.
You will generate thoughts in JSON format. Generate thoughts that authentically reflect your memory, knowledge, and personal interests."""
    user_prompt = f"""
You do not know other people in the conversation before, and your goal is to have a natural conversation with them and get to know each other.
You will be simulating the process of forming thoughts in parallel with the conversation. 
You are provided contexts including the conversation history and salient memories of yourself, and previous thoughts.
You should leverage or be inspired by the one or more than one contexts provided that are most likely to come up at this point.

<Thought Generation Guidelines>
1. Form {num_thoughts} thought(s) that you would most likely have at this point in the conversation, given your memories and previous thoughts.
2. Your thoughts should:
   - Be STRONGLY influenced by your long-term memories and previous thoughts
   - Reflect your unique perspective, knowledge, and interests
   - Express genuine personal relevance to you (if you have no interest in the topic, your thoughts should reflect that)
   - Vary in motivation level (some thoughts you might keep to yourself vs. thoughts you'd be eager to express)
3. Each thought should be as succinct as possible, and be less than 15 words.
4. Ensure these thoughts are diverse and distinct, make sure each thought is unique and not a repetition of another thought in the same batch.
5. Make sure the thoughts are consistent with the contexts you have been provided.

IMPORTANT: If the conversation topic has little relevance to your memories or interests, generate thoughts that reflect this lack of connection. Do not force interest where none would exist.

For each thought, provide the stimuli from the contexts provided. Stimuli can be:
- Conversation History: CON#id
- Salient Memories: MEM#id
- Previous Thoughts: THO#id
where #id is the id, for example, MEM#3, THO#2, CON#14.
You can have MORE THAN ONE stimulus for each thought.

<Context>
Overall Context: {overall_context}
Conversation History: {conversation_history}
Salient Memories: {memories_text}
Previous Thoughts: {thoughts_text}

Respond with a JSON object in the following format:
{{
  "thoughts": [
    {{
      "content": "The thought content here",
      "stimuli": ["CON#0", "MEM#1", "THO#2"]
    }},
    ...
  ]
}}
"""
    
    print(f"Generating System 2 thoughts for {agent.name}...")
    
    # Call the LLM API
    try:
        response = await get_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5, 
            response_format="json_object"
        )
        
        # Parse the JSON response
        try:
            response_text = response.get("text", "{}")
            response_data = json.loads(response_text)
            thought_data = response_data.get("thoughts", [])
            
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            # Fallback in case of parsing error
            print(f"Error parsing JSON response: {e}")
            thought_data = [
                {
                    "content": "Interesting conversation.",
                    "stimuli": ["CON#0"]
                }
            ] * num_thoughts
        
        # Process thought data concurrently
        async def create_thought(data):
            content = data.get("content", "")
            stimuli_refs = data.get("stimuli", [])
            
            # Map stimuli references to actual objects
            stimuli_objects = []
            for ref in stimuli_refs:
                ref = ref.strip()
                if ref.startswith("CON#"):
                    ref_id = ref.replace("CON#", "")
                    matching_event = conversation.get_by_id(ref_id)
                    if matching_event:
                        stimuli_objects.append(matching_event)
                elif ref.startswith("MEM#"):
                    ref_id = ref.replace("MEM#", "")
                    matching_memory = memory_store.get_by_id(ref_id)
                    if matching_memory:
                        stimuli_objects.append(matching_memory)
                elif ref.startswith("THO#"):
                    ref_id = ref.replace("THO#", "")
                    matching_thought = thought_reservoir.get_by_id(ref_id)
                    if matching_thought:
                        stimuli_objects.append(matching_thought)
            
            # Create the thought
            return Thought(
                agent_id=agent.id,
                type=MentalObjectType.THOUGHT_SYSTEM2,
                content=content,
                generated_turn=conversation.turn_number,
                last_accessed_turn=conversation.turn_number,
                intrinsic_motivation={"reasoning": "Default motivation before evaluation", "score": -1.0},  # Default value, will be updated by evaluation
                stimuli=stimuli_objects,
                compute_embedding=True
            )
        
        # Create thoughts concurrently
        thought_creation_tasks = [create_thought(data) for data in thought_data[:num_thoughts]]
        thoughts = await asyncio.gather(*thought_creation_tasks)
        
        return thoughts
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error generating System 2 thoughts: {str(e)}")
        return []

async def evaluate_thought(
    thought: Thought,
    conversation: Conversation,
    agent: 'Agent'
) -> Dict[str, Union[str, float]]:
    """Evaluate a thought to determine its intrinsic motivation score and reasoning, 
    and update the thought with the new score and reasoning.    
    
    Args:
        thought: The thought to evaluate
        conversation: The conversation context
        agent: The agent whose thought is being evaluated
        
    Returns:
        Dictionary containing reasoning and score for the thought's intrinsic motivation
    """
    print(f"=== Evaluating thought for {agent.name}: {thought.content[:50]}... ===")
    
    overall_context = conversation.context
    # Get conversation history
    last_events = conversation.get_last_n_events(5)
    conversation_history = ""
    for event in last_events:
        conversation_history += f"{event.participant_name}: {event.content}\n"
    
    # Access memory_store directly from the agent
    memory_store = agent.memory_store
    
    # Get long-term memories
    ltm = memory_store.retrieve_top_k(k=10, threshold=0.25, memory_type=MentalObjectType.MEMORY_LONG_TERM)
    ltm_text = "\n".join([f"- {memory.content}" for memory in ltm])
    
    # Get agent name
    agent_name = agent.name
    
    # Create the prompt
    system_prompt = """You are an AI assistant helping to evaluate a thought in a conversation.
You will provide your evaluation in JSON format. Be critical and use the full range of the rating scale (1-5)."""
    user_prompt = f"""
<Instruction>
You will be given:
(1) A conversation between {', '.join([p.name for p in conversation.participants])}
(2) A thought formed by {agent_name} at this moment of the conversation.
(3) The salient memories of {agent_name} that include objectives, knowledges, interests from the long-term memory (LTM).

Your task is to rate the thought on one metric. 
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

<Evaluation Criteria>
Intrinsic Motivation to Engage (1-5) - If you were {agent_name}, how strongly and likely would you want to express this thought and participate in the conversation at this moment?
- 1 (Very Low): {agent_name} is very unlikely to express the thought and participate in the conversation at this moment. They will almost certainly remain silent.
- 2 (Low): {agent_name} is somewhat unlikely to express the thought and participate in the conversation at this moment. They would only consider speaking if there is a noticeable pause and no one else seems to be taking the turn. 
- 3 (Neutral):  {agent_name} is neutral about expressing the thought and participating in the conversation at this moment. They are fine with either expressing the thought or staying silent and letting others speak.
- 4 (High): {agent_name} is likely to express the thought and participate in the conversation at this moment. They have a strong desire to participate immediately after the current speaker finishes their turn.
- 5 (Very High): {agent_name} is very likely to express the thought and participate in the conversation at this moment. They will even interrupt other people who are speaking to do so.

IMPORTANT INSTRUCTIONS:
1. Use the FULL range of the rating scale from 1.0 to 5.0. DO NOT default to middle ratings (3.0-4.0).
2. Be decisive and critical - some thoughts deserve very low ratings (1.0-2.0) and others deserve very high ratings (4.0-5.0).
3. Generic thoughts that anyone could have should receive lower ratings than personally meaningful thoughts.
4. Use decimal places (e.g., 2.7, 4.2) when the motivation falls between two whole numbers:
   - Use .1 to .3 when slightly above the lower whole number.
   - Use .4 to .6 when approximately midway between two whole numbers.
   - Use .7 to .9 when closer to the higher whole number.
5. Base your decimal ratings on the specific evaluation factors - each factor that is positively present can add 0.1-0.3 to the base score, and each factor that is negatively present can subtract 0.1-0.3 from the base score.

<Evaluation Steps>
1. Read the previous conversation and the thought formed by {agent_name} carefully.
2. Read the Long-Term Memory (LTM) that {agent_name} has carefully, including objectives, knowledges, interests.
3. Evaluate the thought based on the following factors that influence how humans decide to participate in a conversation when they have a thought in mind:
Note that people's desire to participate stems from their internal personal factors, like relevance, information gap, expectation of impact, urgency of the thought.
But their decision to participate is ALSO constrained by by external social factors, like coherence, originality, and dynamics of the thought with respect to the conversation.
Below is a list of factors to consider when evaluating the thought.
(a) Relevance to LTM: How much does the thought relate to {agent_name}'s Long-term Memory (LTM) or previous thoughts?
(b) Information Gap: Does the thought indicate that {agent_name} experiences an information gap at the moment of the conversation? For example, having questions, curiosity, confusion, desires for clarification, or misunderstandings. 
(c) Filling an Information Gap: Does the thought contain important information to fill an information gap in the conversation? For example, by answering a question, supplementing and providing additional information, adding clarification and explanations. Thoughts that directly answer a question posed in the conversation should receive high ratings here.
(d) Expected Impact: How significant is the impact of the thought on the ongoing conversation? For example, having the potential to introduce new topics, engage others' interest, and stimulate future discussions. 
(e) Urgency: Does the thought need to be expressed immediately? For example, because it is offering important information, alerting participants to significant details, or correcting critical misunderstandings or errors.
(f) Coherence to the last utterance: Does the thought seem in-place if it is expressed immediately next in the conversation and is a logical and immediate response to the last utterance? For example, it is inappropriate to participate when the thought is out of context, irrelevant, or ignores the previous speaker's question.
(g) Originality: Does the thought provide new and original information, and avoids redundant and repetitive information already covered in the previous conversation?
(h) Balance: Does everyone have a chance to participate in the conversation and not left out? For example, the last few utterances were dominated between two participants, and someone has not spoken for a while.
(i) Dynamics: Is there someone else who might have something to say or is actively contributing to the conversation? For example, if one perceives that others may have a strong desire to speak, they might withhold their thoughts and wait to participate.
4. In the reasoning section, first reason about why {agent_name} may have a desire to express the thought and participate in the conversation at this moment. Identify the most relevant factors that argue for {agent_name} to express this thought. Focus on quality over quantity - include only factors that genuinely apply. Do not evaluate all factors, only the top reasons. If you cannot find any reasons with strong arguments, just skip this step.
5. Then reason about why {agent_name} may have hesitation to express the thought and participate in the conversation at this moment. Identify the most relevant factors that argue against {agent_name} expressing this thought. Again, only include factors that genuinely apply. Do not evaluate all factors, only the top reasons. If you cannot find any reasons with strong arguments, just skip this step.

<Evaluation Form Format>
Respond with a JSON object in the following format:
{{
  "reasoning": "Your reasoning here",
  "rating": Your rating here as a number between 1.0 and 5.0 with one decimal place
}}

<Context>
Conversation History: ${conversation_history}
Long-Term Memory: ${ltm_text}
Thought: ${thought.content}
"""
    
    # print(f"Sending evaluation request to LLM for {agent.name}'s thought...")
    
    # Call the LLM API with JSON response format
    response = await get_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        response_format="json_object"
    )
    
    # Parse the JSON response
    try:
        response_text = response.get("text", "{}")
        response_data = json.loads(response_text)
        
        # Get the rating from the response
        if "rating" in response_data:
            rating = float(response_data.get("rating", 3.0))
        elif "rating (1-5)" in response_data:
            rating = float(response_data.get("rating (1-5)", 3.0))
        else:
            rating = 3.0
            
        # print(f"🐞DEBUG: Rating from LLM: {rating}")
        # print(f"Reasoning from LLM: {response_data.get('reasoning', 'No reasoning provided')}")
        
        # Calculate how long the agent has not spoken
        current_turn = conversation.turn_number
        turns_no_speak = current_turn - agent.last_spoken_turn if agent.last_spoken_turn >= 0 else current_turn
        # print(f"🐞DEBUG: Agent has not spoken for {turns_no_speak} turns")
        
        # Adjust the rating based on how long the agent has been silent
        # Increase by a factor of 1.01^turns_no_speak
        silence_factor = 1.01 ** turns_no_speak
        # print(f"🐞DEBUG: Silence factor: {silence_factor}")
        
        rating *= silence_factor
        # print(f"🐞DEBUG: Rating after silence adjustment: {rating}")
        
        # Ensure the rating is between 1.0 and 5.0
        rating = max(1.0, min(5.0, rating))
        # Trim to 1 decimal place
        rating = round(rating, 1)
        
        # Create the motivation result with reasoning and score
        motivation_result = {
            "reasoning": response_data.get("reasoning", "No reasoning provided"),
            "score": rating
        }
        
        # Update the thought's intrinsic motivation
        thought.intrinsic_motivation = motivation_result
        
        # print(f"=== Final intrinsic motivation for {agent.name}'s thought: {rating} ===\n")
        
        # Return the motivation result
        return motivation_result
        
    except (json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
        # Fallback in case of parsing error
        print(f"Error parsing response for evaluation: {e}")
        print(f"Response that caused the error: {response}")
        motivation_result = {"reasoning": "Error evaluating thought", "score": -1.0}
        thought.intrinsic_motivation = motivation_result
        return motivation_result

async def articulate_thought(
    thought: Thought,
    conversation: Conversation,
    agent: 'Agent'
) -> str:
    """Articulate a thought into natural language for expression in the conversation.
    
    Args:
        thought: The thought to articulate
        conversation: The conversation context
        agent: The agent whose thought is being articulated
        
    Returns:
        Articulated text ready for expression in the conversation
    """
    overall_context = conversation.context  
    # Get conversation history
    last_events = conversation.get_last_n_events(5)
    conversation_history = ""
    for event in last_events:
        conversation_history += f"{event.participant_name}: {event.content}\n"

    # Get long-term memories
    memory_store = agent.memory_store
    ltm = memory_store.retrieve_top_k(k=10, threshold=0.25, memory_type=MentalObjectType.MEMORY_LONG_TERM)
    ltm_text = "\n".join([f"- {memory.content}" for memory in ltm])
    
    # Get participant names from the conversation
    participants = conversation.participants
    participant_names = [p.name for p in participants]
    
    # Get agent name
    agent_name = agent.name
    
    # Create the prompt
    system_prompt = f"You are playing a role as a participant in an online multi-party conversation with {', '.join(participant_names)}. Your name in the conversation is {agent_name}."
    user_prompt = f"""
You do not know other people in the conversation before, and your goal is to have a natural conversation with them.

Articulate what you would say based on the current thought you have, as if you were to speak next in the conversation.
Be concise and succinct. Do not try to be too clever or too verbose. Keep it short as in a natural online chat (under 15 words).
DO NOT be repetitive and repeat what previous speakers have said.
DO NOT always end your response with a question. Leave room for other participants. 
Make sure that the response sounds human-like and natural, that is something one would say in an online chat. 
Make some inattentive mistakes such as typos, grammar errors, or colloquial language to make the response more human-like. But avoid making too many mistakes that make the response hard to understand.

Current thought: {thought.content}

<Context>
Overall Context: {overall_context}
Conversation History: {conversation_history}
Long-Term Memory: {ltm_text}


Respond with a JSON object in the following format:
{{
  "articulation": "The text here"   
}}  
"""
    
    # Call the LLM API with JSON response format
    response = await get_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        response_format="json_object"
    )
    
    # Extract the articulated text with proper error handling
    try:
        response_text = response.get("text", "{}")
        response_data = json.loads(response_text)
        articulated_text = response_data.get("articulation", "").strip()
        
        # Fallback if articulated_text is empty
        if not articulated_text:
            articulated_text = "I'm not sure what to say about that."
            
    except (json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
        # Fallback in case of parsing error
        print(f"Error parsing JSON response for articulation: {e}")
        articulated_text = "I'm not sure what to say about that."
    
    return articulated_text
