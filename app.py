import streamlit as st
import asyncio
from src.LLM import GroqLLaMa  # Import your class

def main():
    st.set_page_config(page_title="Academic AI Assistant", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Student Profile", "Study Planner", "AI Assistant"])
    
    if page == "Home":
        home_page()
    elif page == "Student Profile":
        student_profile()
    elif page == "Study Planner":
        study_planner()
    elif page == "AI Assistant":
        ai_assistant()

def home_page():
    st.title("📚 Academic AI Assistant")
    st.write("Welcome to your personalized academic assistant. Navigate through the sidebar to explore different features.")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    if api_key:
        llm = GroqLLaMa(api_key)

        # User input
        user_input = st.chat_input("Enter your prompt:")
            
        # if st.button("Generate Response"):
        if user_input:
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant"):
                st.write("Generating response...")
                async def generate_response():
                    messages = [{"role": "user", "content": user_input}]
                    response = await llm.agenerate(messages)
                    return response

                response = asyncio.run(generate_response())

                # Display Response
                st.subheader("Generated Response:")
                st.write(response)
    
def student_profile():
    st.title("📝 Student Profile Setup")
    name = st.text_input("Enter your name")
    learning_style = st.selectbox("Preferred Learning Style", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
    study_hours = st.slider("Daily Study Hours", 1, 10, 3)
    if st.button("Save Profile"):
        st.success(f"Profile Saved for {name} with {learning_style} learning style.")


def study_planner():
    st.title("📅 Study Planner")
    st.write("Generate your personalized study schedule here.")
    subjects = st.text_area("Enter subjects/topics (comma-separated)")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    if st.button("Generate Schedule"):
        st.success("Study schedule generated successfully!")


def ai_assistant():
    st.title("🤖 AI Assistant")
    st.write("Ask questions or get personalized study recommendations.")
    user_query = st.text_area("Enter your academic query")
    if st.button("Get AI Response"):
        st.info("AI Response: Feature under development.")

if __name__ == "__main__":
    main()
