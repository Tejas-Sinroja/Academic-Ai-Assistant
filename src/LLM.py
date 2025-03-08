import groq
import asyncio
from langchain_groq import ChatGroq
from typing import List, Dict, Optional

class LLMConfig:
    """Configuration settings for the LLM."""
    model: str = "llama3-70b-8192"  # Updated to use Groq's llama3 model
    max_tokens: int = 1024
    default_temp: float = 0.5

class GroqLLaMa:
    """
    A class to interact with Groq Cloud API using ChatGroq.
    """

    def __init__(self, api_key: str):
        """Initialize GroqLLaMa with API key.

        Args:
            api_key (str): Groq Cloud API authentication key
        """
        self.config = LLMConfig()
        self.client = ChatGroq(api_key=api_key)
        self.groq_client = groq.Client(api_key=api_key)

    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Generate text using Groq's model via ChatGroq.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        # Use the groq client directly for async operation since ChatGroq doesn't have achat_completion
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.groq_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.default_temp,
                max_tokens=self.config.max_tokens
            )
        )
        
        return response.choices[0].message.content
    
    def generate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Synchronous version of generate for non-async contexts.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        response = self.groq_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.default_temp,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
