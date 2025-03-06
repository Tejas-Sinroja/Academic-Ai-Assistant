from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import List, Dict, Optional

class LLMConfig:
    """Configuration settings for the LLM."""
    model: str = "deepseek-r1-distill-llama-70b"
    max_tokens: int = 1024
    default_temp: float = 0.5

class GroqLLaMa:
    """
    A class to interact with Groq Cloud API using LangChain's ChatGroq client.
    """

    def __init__(self, api_key: str):
        """Initialize GroqLLaMa with API key.

        Args:
            api_key (str): Groq Cloud API authentication key
        """
        self.config = LLMConfig()
        self.client = ChatGroq(model=self.config.model, groq_api_key=api_key)

    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Generate text using Groq's model via LangChain.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            else:
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        response = await self.client.agenerate(
            messages=[langchain_messages],  # LangChain expects a list of lists
            temperature=temperature or self.config.default_temp,
            max_tokens=self.config.max_tokens
        )
        
        return response.generations[0][0].text
