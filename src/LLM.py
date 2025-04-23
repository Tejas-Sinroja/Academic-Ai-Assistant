from openai import OpenAI
import asyncio
from typing import List, Dict, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class LLMConfig:
    """Configuration settings for the LLM."""
    model: str = "meta-llama/llama-4-scout:free"  # Updated to use OpenRouter model
    # max_tokens: int = 1024
    # default_temp: float = 0.5

class OpenRouterLLM:
    """
    A wrapper class for OpenRouter via OpenAI SDK.
    This class provides access to OpenRouter models.
    """

    def __init__(self, api_key: str):
        """Initialize OpenRouterLLM with API key.

        Args:
            api_key (str): OpenRouter API key
        """
        self.config = LLMConfig()
        # Create OpenAI client configured for OpenRouter
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Generate text using OpenRouter's model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        # Use the OpenAI client configured for OpenRouter
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                # temperature=temperature or self.config.default_temp,
                # max_tokens=self.config.max_tokens
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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            # temperature=temperature or self.config.default_temp,
            # max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
