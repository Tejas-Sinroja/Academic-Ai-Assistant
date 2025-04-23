# LLM Integration: OpenRouter

This document outlines how Academic AI Assistant integrates with OpenRouter to provide access to powerful language models for AI-powered features.

## LLM Integration Strategy

Academic AI Assistant **exclusively** uses OpenRouter for all LLM capabilities. The application does not depend directly on OpenAI or other LLM providers, instead accessing models through OpenRouter's unified API.

## Why OpenRouter?

- **Model Flexibility**: OpenRouter provides access to multiple state-of-the-art models through a single API
- **Cost Management**: OpenRouter offers competitive pricing and efficient routing options
- **Powerful Models**: Access to a variety of high-quality models like Llama, Claude, and more
- **Unified Integration**: Single API implementation to access multiple model providers

## Implementation

The application uses two approaches to working with OpenRouter:

1. **OpenAI SDK**: For both synchronous and asynchronous completions via OpenRouter's OpenAI-compatible API
2. **LangChain Integration**: For higher-level abstractions in RAG and other components

## Setting Up Your OpenRouter API Key

To use the application, you'll need an OpenRouter API key:

1. Sign up for an account at [openrouter.ai](https://openrouter.ai)
2. Generate an API key in your dashboard
3. Add it to your `.env` file: `OPENROUTER_API_KEY=your_api_key_here`

## Troubleshooting

If you're having issues with the LLM integration:

1. Verify your OpenRouter API key is correctly set in the `.env` file
2. Ensure you have internet connectivity to reach OpenRouter's API
3. Ensure you have the required packages installed: `openai` and any LangChain components

## Code Structure

The main class in `src/LLM.py` is `OpenRouterLLM`, which provides two primary methods:

- `generate(messages)`: Synchronous completion generation
- `agenerate(messages)`: Asynchronous completion generation

## Example Usage

```python
from src.LLM import OpenRouterLLM

# Initialize the LLM
llm = OpenRouterLLM(api_key="your_openrouter_api_key")

# Generate a completion
messages = [{"role": "user", "content": "Explain the quantum tunneling effect briefly."}]
response = llm.generate(messages)
print(response)
``` 