"""OpenAI API interaction functions."""
import os
from typing import List, Dict, Optional, Any
from openai import OpenAI, APIError
import logging
import asyncio
import time
from openai import AzureOpenAI
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Default model settings that can be overridden through environment variables
DEFAULT_COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "oar-gpt-4o-0806-htpm")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

class LLMAPIError(Exception):
	"""Custom exception for LLM API errors."""
	pass

def get_client() -> AzureOpenAI:
	"""Get OpenAI client with API key validation.
	
	Returns:
		OpenAI client instance
		
	Raises:
		LLMAPIError: If OPENAI_API_KEY is not set
	"""
	# api_key = os.getenv("OPENAI_API_KEY")

	api_key = os.getenv("AOAI_EAST_US_KEY")

	if not api_key:
		raise LLMAPIError("OPENAI_API_KEY environment variable not set")

	return AzureOpenAI(
					# this is the AOAI-east-us endpoint
					azure_endpoint = os.getenv("AOAI_EAST_US_ENDPOINT"),
					api_key = api_key,
					api_version="2024-10-21"
				)

async def get_completion(
	system_prompt: str,
	user_prompt: str,
	model: str = DEFAULT_COMPLETION_MODEL,
	temperature: float = 1.0,
	max_tokens: Optional[int] = None,
	max_retries: int = 3,
	response_format: Optional[str] = None
) -> Dict[str, Any]:
	"""Get completion from OpenAI API.
	
	Args:
		system_prompt: System message
		user_prompt: User message/query
		model: Model to use (default: from environment variable or gpt-4o)
		temperature: Sampling temperature (default: 1.0)
		max_tokens: Maximum tokens in response (optional)
		max_retries: Maximum number of retries on API error (default: 3)
		response_format: Response format (optional, e.g., "json_object")
		
	Returns:
		Dictionary with API response, containing 'text', 'finish_reason', etc.
		
	Raises:
		LLMAPIError: If API call fails after max_retries
	"""
	client = get_client()
	for attempt in range(max_retries):
		try:
			completion_args = {
				"model": model,
				"messages": [
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt}
				],
				"temperature": temperature,
				"max_tokens": max_tokens,
				"top_p": 1.0
			}
			
			if response_format:
				completion_args["response_format"] = {"type": response_format}
			
			response = client.chat.completions.create(**completion_args)
			
			result = {"text": response.choices[0].message.content}
			return result
			
		except APIError as e:
			if e.status_code == 429:  # Rate limit error
				if attempt < max_retries - 1:
					wait_time = 2 ** attempt  # Exponential backoff
					logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
					await asyncio.sleep(wait_time)
					continue
			logger.error(f"OpenAI API error: {str(e)}")
			raise LLMAPIError(f"OpenAI API error: {str(e)}")
			
		except Exception as e:
			logger.error(f"Unexpected error: {str(e)}")
			raise LLMAPIError(f"Unexpected error: {str(e)}")
	
	raise LLMAPIError("Max retries exceeded")


# Async version of the embedding function
async def get_embedding_async(
	text: str,
	model: str = DEFAULT_EMBEDDING_MODEL,
	max_retries: int = 3
) -> List[float]:
	"""Get embedding asynchronously from OpenAI API.
	
	Args:
		text: Text to get embedding for
		model: Model to use (default: from environment variable or text-embedding-3-small)
		max_retries: Maximum number of retries on API error (default: 3)
		
	Returns:
		List of embedding values
		
	Raises:
		LLMAPIError: If API call fails after max_retries
	"""
	if not text.strip():
		raise ValueError("Empty text provided for embedding")
		
	client = get_client()
	for attempt in range(max_retries):
		try:
			response = client.embeddings.create(
				model=model,
				input=text
			)
			return response.data[0].embedding
			
		except APIError as e:
			if e.status_code == 429:  # Rate limit error
				if attempt < max_retries - 1:
					wait_time = 2 ** attempt  # Exponential backoff
					logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
					await asyncio.sleep(wait_time)
					continue
			logger.error(f"OpenAI API error: {str(e)}")
			raise LLMAPIError(f"OpenAI API error: {str(e)}")
			
		except Exception as e:
			logger.error(f"Unexpected error: {str(e)}")
			raise LLMAPIError(f"Unexpected error: {str(e)}")
	
	raise LLMAPIError("Max retries exceeded")

# Synchronous version of the embedding function
def get_embedding_sync(
	text: str,
	model: str = DEFAULT_EMBEDDING_MODEL,
	max_retries: int = 3
) -> List[float]:
	"""Get embedding synchronously from OpenAI API.
	
	Args:
		text: Text to get embedding for
		model: Model to use (default: from environment variable or text-embedding-3-small)
		max_retries: Maximum number of retries on API error (default: 3)
		
	Returns:
		List of embedding values
		
	Raises:
		LLMAPIError: If API call fails after max_retries
	"""
	if not text.strip():
		raise ValueError("Empty text provided for embedding")
		
	client = get_client()
	for attempt in range(max_retries):
		try:
			response = client.embeddings.create(
				model=model,
				input=text
			)
			return response.data[0].embedding
			
		except APIError as e:
			if e.status_code == 429:  # Rate limit error
				if attempt < max_retries - 1:
					wait_time = 2 ** attempt  # Exponential backoff
					logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
					# Use time.sleep instead of asyncio.sleep for synchronous code
					time.sleep(wait_time)
					continue
			logger.error(f"OpenAI API error: {str(e)}")
			raise LLMAPIError(f"OpenAI API error: {str(e)}")
			
		except Exception as e:
			logger.error(f"Unexpected error: {str(e)}")
			raise LLMAPIError(f"Unexpected error: {str(e)}")
	
	raise LLMAPIError("Max retries exceeded") 
