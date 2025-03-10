import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from inner_thoughts_ai.models.conversation import Conversation, Event
    from inner_thoughts_ai.models.participant import Agent

# Import directly to fix the NameError
from inner_thoughts_ai.models.participant import Participant
from inner_thoughts_ai.models.conversation import Conversation

from inner_thoughts_ai.utils.llm_api import get_completion

async def predict_turn_taking(conversation: 'Conversation') -> str:
    """Predict turn taking type based on the last 5 utterances.
    
    Args:
        conversation: The conversation to predict the turn taking type for
        
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


async def decide_next_speaker_and_utterance(conversation: 'Conversation') -> Tuple[Optional[Participant], str]:
    """Decide the next speaker and their utterance based on current conversation state.
    Get all selected thoughts from all participants, and then select the one with the highest intrinsic motivation score.
    """
    # Get all selected thoughts from all participants
    selected_thoughts = []
    for participant in conversation.participants:
        selected_thoughts.extend(participant.thought_reservoir.get_selected_thoughts())

    if len(selected_thoughts) == 0:
        return None, None
    
    # Select the thought with the highest intrinsic motivation score
    selected_thought = max(selected_thoughts, key=lambda x: x.intrinsic_motivation['score'])

    participant = conversation.get_participant_by_id(selected_thought.agent_id)
    # Articulate the thought
    from inner_thoughts_ai.utils.thinking_engine import articulate_thought
    utterance = await articulate_thought(selected_thought, conversation, agent=participant)
    
    # Return the next speaker and their utterance
    return participant, utterance


    
    
    