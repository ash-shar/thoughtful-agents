from typing import Dict, List
import asyncio

from inner_thoughts_ai.models.conversation import Conversation, Event
from inner_thoughts_ai.utils.llm_api import get_completion

class TurnTakingManager:
    async def pred_turn_taking(self, conversation: Conversation) -> str:
        """Predict turn taking based on the conversation.
        
        Args:
            conversation: The conversation to predict turn taking for
            
        Returns:
            The predicted next speaker's name or "anyone" if uncertain
        """
        # Get the last 5 events from the conversation
        last_events = conversation.get_last_n_events(5)
        
        # Extract participant information
        participants = conversation.get_participants()
        participant_list = ", ".join([p.name for p in participants])
        num_participants = len(participants)
        
        # Format the last 5 utterances
        last_5_utterances = ""
        for event in last_events:
            # Find the participant name based on participant_id
            speaker_name = "Unknown"
            for p in participants:
                if p.id == event.participant_id:
                    speaker_name = p.name
                    break
            
            last_5_utterances += f"{speaker_name}: {event.content}\n"
        
        # Create the prompts
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
                
            return predicted_speaker
            
        except Exception as e:
            # Log the error and default to "anyone"
            print(f"Error predicting turn taking: {str(e)}")
            return "anyone"
    
    
    def decide_speaker(self, conversation: 'Conversation') -> Dict[str, str]:
        """Decide the next speaker and their utterance."""
        pass