import os
import streamlit as st
from openai import OpenAI
from streamlit_markmap import markmap

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="AI-Powered Mind-Map Generator", layout="wide")
st.title("🔗 Academic AI Assistant: Mind-Map Generator")

# ── Initialize OpenRouter client via OpenAI SDK ────────────────────────────────
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# if not OPENROUTER_API_KEY:
#     st.error("Please set the OPENROUTER_API_KEY environment variable.")
#     st.stop()

# client = OpenAI(
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1",   # Route OpenAI calls to OpenRouter :contentReference[oaicite:5]{index=5}
# )

# ── User input ────────────────────────────────────────────────────────────────
prompt = st.text_area(
    label="Enter text to convert into a mind-map outline",
    value="Reinforcement Learning studies how agents learn by interacting with environments. Key concepts include Markov decision processes, Q-learning, policy gradients, and exploration–exploitation trade-offs.",
    height=200
)

if st.button("Generate Mind-Map"):
    # ── Ask the LLM for a Markdown mind-map outline ────────────────────────────
    messages = [
        {"role": "system", "content": "You are a tool that converts academic text into mind maps."},
        {"role": "user",   "content": f"Convert the following text into a Markdown mind-map outline:\n\n{prompt}"}
    ]
    # response = client.chat.completions.create(
    #     model="openrouter/gpt-3.5-turbo",    # Or any OpenRouter-accessible model :contentReference[oaicite:6]{index=6}
    #     messages=messages,
    #     temperature=0.2,
    #     max_tokens=512,
    # )
    # ── Static Markdown mind-map for testing ─────────────────────────────────────────
    md_mindmap = """
        - Academic AI Assistant
        - Inputs
            - PDF Upload
            - Plain Text Prompt
        - Processing Steps
            - Text Extraction
            - Hierarchy Structuring
            - Mind-Map Formatting
        - Outputs
            - Narrative Summary
            - Interactive Mind-Map
        - Tech Stack
            - Streamlit
            - OpenRouter LLM
            - markmap.js
        - UI Components
            - Text Area
            - Code Block Display
            - Mind-Map Canvas
        """

    
    # ── Display the raw Markdown (optional) ────────────────────────────────────
    st.markdown("**Generated Markdown Mind-Map:**")
    st.code(md_mindmap, language="markdown")
    
    # ── Render as interactive mind-map ───────────────────────────────────────
    st.markdown("**Interactive Mind-Map:**")
    markmap(md_mindmap, height=600)          # Uses streamlit-markmap under the hood :contentReference[oaicite:7]{index=7}

# ── Footer & links ────────────────────────────────────────────────────────────
st.markdown("---")
# st.write("More OpenRouter examples: https://github.com/OpenRouterTeam/openrouter-examples")  :contentReference[oaicite:8]{index=8}
