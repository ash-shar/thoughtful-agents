import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from inner_thoughts_ai.models.conversation import Conversation, Event
    from inner_thoughts_ai.models.participant import Participant, Agent

from inner_thoughts_ai.utils.llm_api import get_completion

async def predict_next_speaker(conversation: 'Conversation') -> str:
    """Predict the next speaker in the conversation.
    
    Args:
        conversation: The conversation to predict the next speaker for
        
    Returns:
        The name of the predicted next speaker, or "anyone" if no clear prediction
    """
    # Get the last 5 utterances
    last_events = conversation.get_last_n_events(5)
    last_5_utterances = ""
    for event in last_events:
        last_5_utterances += f"{event.participant_name}: {event.content}\n"
    
    # Get the participants
    participants = conversation.participants
    num_participants = len(participants)
    participant_list = ", ".join([p.name for p in participants])
    
    # Create the prompt
    system_prompt = f"This is a conversation between {num_participants} speakers. The speakers are: {participant_list}. Predict who the next speaker will be based on the last 5 utterances. Return ONLY the speaker name. If the next speaker is not clearly allocated to a specific speaker and any speaker could take the floor in the next turn, return \"anyone\"."
    user_prompt = f"<Task>Last 5 utterances:\n{last_5_utterances}\nPrediction: "
    
    # Call the OpenAI API
    try:
        response = await get_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,  # Lower temperature for more deterministic results
        )
        
        # Extract the predicted speaker from the response
        predicted_speaker = response.get("text", "").strip()
        
        # Validate the prediction
        valid_speakers = [p.name for p in participants] + ["anyone"]
        if predicted_speaker not in valid_speakers:
            # If the prediction is not valid, default to "anyone"
            return "anyone"
        
        # Update the last event with the predicted next speaker
        if last_events and predicted_speaker != "anyone":
            last_event = last_events[-1]
            last_event.pred_next_turn = predicted_speaker
        
        return predicted_speaker
        
    except Exception as e:
        # Log the error and default to "anyone"
        print(f"Error predicting turn taking: {str(e)}")
        return "anyone"


async def decide_next_speaker(conversation: 'Conversation') -> Tuple[Optional['Participant'], str]:
    """Decide the next speaker and their utterance based on current conversation state.
    
    Args:
        conversation: The conversation to analyze
        
    Returns:
        A tuple of (participant, utterance) where participant may be None if no decision
    """
    # Dictionary to collect thoughts from all agents
    all_agent_thoughts = {}
    
    # Gather all agent thoughts concurrently using asyncio.gather
    agents = [p for p in conversation.participants if hasattr(p, 'act')]
    
    # Create a list of coroutines to run concurrently
    act_coroutines = [agent.act(conversation) for agent in agents]
    
    # Run them all concurrently
    results = await asyncio.gather(*act_coroutines)
    
    # Store the results
    for agent, thoughts in zip(agents, results):
        if thoughts:
            all_agent_thoughts[agent] = thoughts
    
    # If no agent has thoughts to articulate, return None
    if not all_agent_thoughts:
        return None, ""
    
    # Find the agent with the highest-rated thought
    best_agent = None
    best_thought = None
    highest_score = -1
    
    for agent, thoughts in all_agent_thoughts.items():
        for thought in thoughts:
            score = thought.intrinsic_motivation.get('score', 0)
            if score > highest_score:
                highest_score = score
                best_thought = thought
                best_agent = agent
    
    # If we found a thought to articulate
    if best_agent and best_thought:
        # Articulate the thought
        utterance = await best_agent.articulate_thought(best_thought, conversation)
        return best_agent, utterance
    
    # No suitable thought found, select a random thought from a random agent
    random_agent = random.choice(list(all_agent_thoughts.keys()))
    random_thought = random.choice(all_agent_thoughts[random_agent])
    utterance = await random_agent.articulate_thought(random_thought, conversation)
    return random_agent, utterance


