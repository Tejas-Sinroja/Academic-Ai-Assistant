import streamlit as st
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from pathlib import Path
import sys
import asyncio
from src.LLM import GroqLLaMa  # Import the LLM class
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import nest_asyncio
import uuid
import validators
import json
import tempfile
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from src.agents.notewriter import get_notewriter
from src.agents.planner import get_planner
from src.agents.advisor import get_advisor
from src.extractors import extract_youtube_id

# RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_qa_with_sources_chain, create_history_aware_retriever, RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Add source directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Connection Settings 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Apply nest_asyncio to allow asyncio to work in Streamlit 
# This enables asynchronous content extraction in the app
nest_asyncio.apply()

def init_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError:
        # Database might not exist yet, connect to default postgres db
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if our DB exists, if not create it
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        if not cursor.fetchone():
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        
        cursor.close()
        conn.close()
        
        # Now connect to our database
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

def init_db():
    """Initialize database tables if they don't exist"""
    conn = init_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE,
        learning_style VARCHAR(50),
        study_hours INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id),
        title VARCHAR(255) NOT NULL,
        description TEXT,
        due_date TIMESTAMP,
        priority VARCHAR(50),
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id),
        title VARCHAR(255) NOT NULL,
        content TEXT,
        subject VARCHAR(100),
        tags TEXT[],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        content TEXT,
        embedding_vector BYTEA,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

def check_db_schema():
    """Check if the database schema is up-to-date and return a list of issues"""
    schema_issues = []
    
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        # Check if source_type column exists in notes table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_type'
        """)
        
        if not cursor.fetchone():
            schema_issues.append("Notes table missing 'source_type' column")
        
        # Check if source_url column exists in notes table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_url'
        """)
        
        if not cursor.fetchone():
            schema_issues.append("Notes table missing 'source_url' column")
            
        # Check if mindmap_content column exists in notes table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'mindmap_content'
        """)
        
        if not cursor.fetchone():
            schema_issues.append("Notes table missing 'mindmap_content' column")
            
        # Check if quizzes table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'quizzes'
            )
        """)
        
        if not cursor.fetchone()[0]:
            schema_issues.append("Quizzes table doesn't exist")
        
        cursor.close()
        conn.close()
    except Exception as e:
        schema_issues.append(f"Error checking schema: {str(e)}")
    
    return schema_issues

def main():
    st.set_page_config(
        page_title="Academic AI Assistant", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_db()
    
    # Check if database schema is up to date
    schema_issues = check_db_schema()
    if schema_issues:
        st.warning("""
        ⚠️ Database schema needs to be updated to enable all features.
        
        Please run the following command to update your database:
        ```
        python update_db_schema.py
        ```
        
        Issues detected:
        - """ + "\n- ".join(schema_issues))
    
    # Sidebar for navigation
    st.sidebar.title("Academic AI Assistant")
    st.sidebar.image("https://img.icons8.com/color/96/000000/student-male--v1.png", width=100)
    
    # Navigation options
    pages = {
        "Home": "🏠",
        "Notewriter": "📝",
        "Planner": "📅",
        "Advisor": "🧠",
        "Quiz & Analyze": "📚",
        "QnA": "💬",
        "Dashboard": "📊"
    }
    
    # Create selection box with icons
    selection = st.sidebar.radio(
        "Navigate to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Display appropriate page based on selection
    if selection == "Home":
        home_page()
    elif selection == "Notewriter":
        notewriter_page()
    elif selection == "Planner":
        planner_page()
    elif selection == "Advisor":
        advisor_page()
    elif selection == "QnA":
        pdf_chat_page()
    elif selection == "Quiz & Analyze":
        quiz_analyze_page()
    elif selection == "Dashboard":
        dashboard_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Academic AI Assistant")
    
def home_page():
    st.title("🏠 Academic AI Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Your Personal Academic Companion
        
        Academic AI Assistant is your all-in-one solution for managing your academic life. 
        Our intelligent agents work together to provide you with personalized support:
        
        - **📝 Notewriter**: Generate study materials and summarize lectures
        - **📅 Planner**: Optimize your schedule and manage your academic calendar
        - **🧠 Advisor**: Get personalized learning and time management advice
        - **📚 Quiz & Analyze**: Test your knowledge and analyze your understanding
        
        Get started by exploring the different features using the sidebar navigation.
        """)
        
        st.info("💡 To begin, set up your student profile using the form on the right.")
    
    with col2:
        st.subheader("Quick Profile Setup")
        with st.form("profile_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Email")
            learning_style = st.selectbox(
                "Learning Style",
                ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
            )
            study_hours = st.slider("Daily Study Hours", 1, 12, 3)
            
            submit = st.form_submit_button("Save Profile")
            
            if submit and name and email:
                conn = init_connection()
                cursor = conn.cursor()
                
                # Check if email already exists
                cursor.execute("SELECT id FROM students WHERE email = %s", (email,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    # Update existing user
                    cursor.execute("""
                        UPDATE students 
                        SET name = %s, learning_style = %s, study_hours = %s
                        WHERE email = %s
                    """, (name, learning_style, study_hours, email))
                    user_id = existing_user[0]
                    st.success("Profile updated successfully!")
                else:
                    # Insert new user
                    cursor.execute("""
                        INSERT INTO students (name, email, learning_style, study_hours)
                        VALUES (%s, %s, %s, %s) RETURNING id
                    """, (name, email, learning_style, study_hours))
                    user_id = cursor.fetchone()[0]
                    st.success("Profile created successfully!")
                
                conn.commit()
                cursor.close()
                conn.close()
                
                # Store user_id in session state
                st.session_state['user_id'] = user_id
                st.session_state['user_name'] = name
    
    # Display system overview
    st.markdown("---")
    st.subheader("🔍 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📝 Notewriter")
        st.markdown("""
        - Generate comprehensive notes
        - Summarize lecture content
        - Create flashcards for study
        """)
        st.button("Go to Notewriter", key="home_to_notewriter", on_click=lambda: st.session_state.update({"navigation": "Notewriter"}))
    
    with col2:
        st.markdown("### 📅 Planner")
        st.markdown("""
        - Optimize your study schedule
        - Manage deadlines and tasks
        - Sync with your calendar
        """)
        st.button("Go to Planner", key="home_to_planner", on_click=lambda: st.session_state.update({"navigation": "Planner"}))
    
    with col3:
        st.markdown("### 🧠 Advisor")
        st.markdown("""
        - Get personalized learning advice
        - Optimize study techniques
        - Receive time management tips
        """)
        st.button("Go to Advisor", key="home_to_advisor", on_click=lambda: st.session_state.update({"navigation": "Advisor"}))
        
    with col4:
        st.markdown("### 📚 Quiz & Analyze")
        st.markdown("""
        - Test your knowledge
        - Generate practice questions
        - Get personalized feedback
        """)
        st.button("Go to Quiz & Analyze", key="home_to_quiz", on_click=lambda: st.session_state.update({"navigation": "Quiz & Analyze"}))

def notewriter_page():
    st.title("📝 Notewriter")
    
    st.markdown("""
    The Notewriter agent helps you process academic content and generate study materials 
    tailored to your learning style.
    """)
    
    # LLM selection and configuration
    st.sidebar.markdown("### 🧠 LLM Configuration")
    llm_type = st.sidebar.radio(
        "Select Language Model Provider:",
        ["Groq", "OpenRouter"],
        index=0,
        help="Choose which LLM provider to use for generating notes. Groq is optimized for speed, OpenRouter provides access to multiple models."
    )
    
    # Handle API keys based on selected LLM
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    if llm_type == "Groq":
        if not groq_api_key or groq_api_key == "your_groq_api_key":
            if "groq_api_key" in st.session_state:
                groq_api_key = st.session_state.groq_api_key
            else:
                groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", key="groq_input")
                if groq_api_key:
                    st.session_state.groq_api_key = groq_api_key
                    st.session_state.api_key = groq_api_key  # For backward compatibility
        
        # Store selected LLM type
        st.session_state.selected_llm_type = "groq"
        st.session_state.selected_api_key = groq_api_key
        
    else:  # OpenRouter
        if not openrouter_api_key or openrouter_api_key == "your_openrouter_api_key":
            if "openrouter_api_key" in st.session_state:
                openrouter_api_key = st.session_state.openrouter_api_key
            else:
                openrouter_api_key = st.sidebar.text_input("Enter OpenRouter API Key:", type="password", key="openrouter_input")
                if openrouter_api_key:
                    st.session_state.openrouter_api_key = openrouter_api_key
        
        # Store selected LLM type
        st.session_state.selected_llm_type = "openrouter"
        st.session_state.selected_api_key = openrouter_api_key
        
        # Show model selection if OpenRouter is selected
        if openrouter_api_key:
            openrouter_model = st.sidebar.selectbox(
                "Select OpenRouter Model:",
                [
                    "deepseek/deepseek-chat-v3-0324:free",
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3-sonnet", 
                    "anthropic/claude-3-haiku",
                    "meta-llama/llama-3-70b-instruct",
                    "google/gemini-1.5-pro"
                    
                ],
                index=0,
                help="Choose which model to use with OpenRouter. Higher-tier models may offer better quality but might be more expensive."
            )
            st.session_state.openrouter_model = openrouter_model
    
    # Source type selection
    st.subheader("Select Content Source")
    source_type = st.radio(
        "Choose your source type:",
        ["Research a Topic", "Text Input", "Web Page", "YouTube Video"],
        horizontal=True
    )
    
    # Create a form for note creation
    with st.form("notewriter_form"):
        title = st.text_input("Note Title")
        subject = st.text_input("Subject")
        
        # Source content based on selected type
        if source_type == "Research a Topic":
            content = None
            source_url = None
            uploaded_file = None
            
            topic = st.text_input("Enter the topic to research:", placeholder="e.g., Quantum Computing, Renaissance Art, Climate Change")
            search_depth = st.radio(
                "Research Depth:",
                ["Ordinary", "Deep Search"],
                horizontal=True,
                help="Deep Search will gather more sources for a more comprehensive result but takes longer"
            )
            
            st.info("""
            The Notewriter agent will automatically:
            1. Search the web for information on this topic
            2. Find relevant YouTube videos with transcripts
            3. Extract and combine the content
            4. Generate comprehensive notes with source citations
            """)
            
        elif source_type == "Text Input":
            content = st.text_area("Enter lecture content, readings, or notes to process", height=300)
            source_url = None
            uploaded_file = None
            topic = None
            search_depth = None
            
        elif source_type == "Web Page":
            content = None
            source_url = st.text_input("Enter webpage URL:")
            st.info("The Notewriter will extract and process content from the webpage.")
            uploaded_file = None
            topic = None
            search_depth = None
            
        elif source_type == "YouTube Video":
            content = None
            source_url = st.text_input("Enter YouTube video URL:")
            if source_url:
                try:
                    video_id = extract_youtube_id(source_url)
                    if video_id:
                        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading video preview: {str(e)}")
            
            st.info("""
            The Notewriter will extract the transcript and process content from the video.
            
            **Note:** This feature requires that the YouTube video has captions/transcripts available.
            Some videos, especially newer ones or those in certain languages, may not have transcripts available.
            """)
            uploaded_file = None
            topic = None
            search_depth = None
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            output_format = st.selectbox(
                "Output Format",
                ["Comprehensive Notes", "Brief Summary", "Mind Map"]
            )
        with col2:
            focus_area = st.text_input("Focus Area (optional)", 
                                       placeholder="E.g., historical context, methodology")
        with col3:
            tags = st.text_input("Tags (comma separated)", 
                                 placeholder="E.g., biology, cells, mitosis")
        
        submit = st.form_submit_button("Process Content")
        
        if submit:
            if 'user_id' not in st.session_state:
                st.warning("Please set up your profile first on the Home page.")
            elif not groq_api_key:
                st.warning("Please enter your Groq API key to enable AI processing.")
            elif source_type == "Text Input" and not content:
                st.warning("Please enter some content to process.")
            elif source_type == "Web Page" and not source_url:
                st.warning("Please enter a webpage URL.")
            elif source_type == "YouTube Video" and not source_url:
                st.warning("Please enter a YouTube video URL.")
            elif source_type == "Research a Topic" and not topic:
                st.warning("Please enter a topic to research.")
            elif not title or not subject:
                st.warning("Please provide both a title and subject for your notes.")
            else:
                # Get student profile to determine learning style
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT learning_style
                    FROM students
                    WHERE id = %s
                """, (st.session_state['user_id'],))
                
                result = cursor.fetchone()
                cursor.close()
                
                learning_style = result[0] if result else "Visual"
                
                # Get LLM selection from session state
                selected_llm_type = st.session_state.get('selected_llm_type', 'groq')
                api_key = st.session_state.get('selected_api_key', '')
                
                # Check if we have the required API key
                if not api_key:
                    if selected_llm_type == 'groq':
                        st.error("Please enter your Groq API key to enable AI processing.")
                    else:
                        st.error("Please enter your OpenRouter API key to enable AI processing.")
                    return
                
                # Initialize the notewriter agent with the selected LLM
                if selected_llm_type == 'groq':
                    notewriter = get_notewriter(llm_type="groq", groq_api_key=api_key)
                else:
                    # Get the selected OpenRouter model
                    openrouter_model = st.session_state.get('openrouter_model', 'anthropic/claude-3-sonnet')
                    notewriter = get_notewriter(llm_type="openrouter", openrouter_api_key=api_key, openrouter_model=openrouter_model)
                
                if not notewriter:
                    if selected_llm_type == 'groq':
                        st.error("Failed to initialize the Notewriter agent. Please check your Groq API key.")
                    else:
                        st.error("Failed to initialize the Notewriter agent. Please check your OpenRouter API key.")
                    return
                
                # Process content using the appropriate method
                with st.spinner("Processing your content with AI - this may take a few moments..."):
                    try:
                        # Handle different source types
                        if source_type == "Text Input":
                            source_data = content
                            source_type_code = "text"
                        elif source_type == "Web Page":
                            source_data = source_url
                            source_type_code = "web"
                        elif source_type == "YouTube Video":
                            source_data = source_url
                            source_type_code = "youtube"
                        elif source_type == "Research a Topic":
                            # Convert search_depth selection to format expected by the function
                            depth = "deep" if search_depth == "Deep Search" else "ordinary"
                            # Use the specialized topic processing method
                            result = asyncio.run(notewriter.process_topic(
                                student_id=st.session_state['user_id'],
                                topic=topic,
                                search_depth=depth,
                                title=title,
                                subject=subject,
                                focus_area=focus_area,
                                tags=tags,
                                learning_style=learning_style
                            ))
                            
                            # Handle result specially since we already have it
                            if result["success"]:
                                st.success(f"Research on '{topic}' completed and notes saved successfully!")
                                
                                # Store the note ID in session state for viewing
                                st.session_state['selected_note_id'] = result["note_id"]
                                
                                # If Mind Map format is selected, generate and show the mindmap
                                if output_format == "Mind Map":
                                    with st.spinner("Generating Mind Map visualization..."):
                                        # Check if we already have a mindmap for this note
                                        existing_mindmap = notewriter.get_mindmap(result["note_id"])
                                        
                                        if existing_mindmap:
                                            mindmap_content = existing_mindmap
                                            st.success("Using previously generated mind map")
                                        else:
                                            # Generate a new mindmap
                                            mindmap_content = notewriter.generate_mindmap(result["content"])
                                            # Save it to the database for future use
                                            notewriter.save_mindmap(result["note_id"], mindmap_content)
                                            st.success("Mind map generated and saved for future use")
                                        
                                        # Display the mindmap
                                        st.subheader("Mind Map Visualization")
                                        st.info("Below is an interactive mind map of your notes. You can expand/collapse branches by clicking on them.")
                                        
                                        # Use the streamlit-markmap component to visualize
                                        from streamlit_markmap import markmap
                                        markmap(mindmap_content, height=600)
                            else:
                                st.error(f"Error: {result['error']}")
                            
                            # Skip the rest of the processing since we've already handled the topic research
                            return
                        
                        # For non-topic sources, continue with regular processing
                        # Try a specific YouTube validation if needed
                        if source_type_code == "youtube":
                            if not validators.url(source_data):
                                st.error(f"Invalid YouTube URL: {source_data}")
                                st.info("Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)")
                                return
                            
                            if "youtube.com" not in source_data and "youtu.be" not in source_data:
                                st.warning(f"URL doesn't appear to be a YouTube link: {source_data}")
                                continue_anyway = st.button("Continue anyway")
                                if not continue_anyway:
                                    return
                        
                        # Process the source
                        result = asyncio.run(notewriter.process_source(
                            student_id=st.session_state['user_id'],
                            source_type=source_type_code,
                            source=source_data,
                            title=title,
                            subject=subject,
                            focus_area=focus_area,
                            tags=tags,
                            learning_style=learning_style
                        ))
                        
                        if result["success"]:
                            st.success(f"Note '{title}' processed and saved successfully!")
                            
                            # Store the note ID in session state for viewing
                            st.session_state['selected_note_id'] = result["note_id"]
                            
                            # If Mind Map format is selected, generate and show the mindmap
                            if output_format == "Mind Map":
                                with st.spinner("Generating Mind Map visualization..."):
                                    # Check if we already have a mindmap for this note
                                    existing_mindmap = notewriter.get_mindmap(result["note_id"])
                                    
                                    if existing_mindmap:
                                        mindmap_content = existing_mindmap
                                        st.success("Using previously generated mind map")
                                    else:
                                        # Generate a new mindmap
                                        mindmap_content = notewriter.generate_mindmap(result["content"])
                                        # Save it to the database for future use
                                        notewriter.save_mindmap(result["note_id"], mindmap_content)
                                        st.success("Mind map generated and saved for future use")
                                    
                                    # Display the mindmap
                                    st.subheader("Mind Map Visualization")
                                    st.info("Below is an interactive mind map of your notes. You can expand/collapse branches by clicking on them.")
                                    
                                    # Use the streamlit-markmap component to visualize
                                    from streamlit_markmap import markmap
                                    markmap(mindmap_content, height=600)
                            else:
                                st.warning("Mind Map generation requires OpenRouter API key. Please add it to your .env file.")
                        else:
                            # Special handling for YouTube extraction failures
                            if source_type_code == "youtube":
                                st.error(f"Failed to extract YouTube content: {result['error']}")
                                
                                # Provide helpful information and alternatives
                                st.warning("""
                                ### ⚠️ YouTube Transcript Extraction Failed
                                
                                This could be due to one of the following reasons:
                                
                                1. **The video doesn't have captions/transcripts available**
                                2. The video might have disabled automatic transcription
                                3. The video might be in a language not supported by the transcript API
                                4. The video might be private or age-restricted
                                
                                ### What you can do:
                                
                                1. **Try a different YouTube video** that has captions
                                2. **Check if the video has captions** (look for the CC button in YouTube's player)
                                3. **Copy the transcript manually** from YouTube:
                                   - Click the "..." button under the video
                                   - Select "Show transcript"
                                   - Copy and paste it as Text Input
                                """)
                                
                                # Option to view available videos with transcripts
                                if st.button("Show Examples of Videos with Transcripts"):
                                    st.markdown("""
                                    ### Examples of Educational Videos with Transcripts:
                                    
                                    - [Khan Academy: Introduction to Limits](https://www.youtube.com/watch?v=riXcZT2ICjA)
                                    - [TED-Ed: The benefits of a bilingual brain](https://www.youtube.com/watch?v=MMmOLN5zBLY)
                                    - [Crash Course: Introduction to Psychology](https://www.youtube.com/watch?v=vo4pMVb0R6M)
                                    - [MIT OpenCourseWare](https://www.youtube.com/c/mitocw/videos)
                                    """)
                            else:
                                st.error(f"Error: {result['error']}")
                            
                            # If extraction failed but we have direct content, still try to save it
                            if source_type == "Text Input":
                                # Save original content
                                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                                
                                note_data = {
                                    "title": title,
                                    "content": content,
                                    "subject": subject,
                                    "tags": tag_list,
                                    "source_type": "text",
                                    "source_url": None
                                }
                                
                                if notewriter.add_note(st.session_state['user_id'], note_data):
                                    st.info("Original content was saved as a note even though AI processing failed.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    # Display saved notes
    st.subheader("Your Notes")
    
    if 'user_id' in st.session_state:
        notewriter = get_notewriter()
        if notewriter:
            notes = notewriter.get_notes(st.session_state['user_id'])
            
            if not notes:
                st.info("You don't have any notes yet. Create one using the form above.")
            else:
                # Create tabs for all notes and filtering by subject
                all_tab, subject_tab, search_tab = st.tabs(["All Notes", "Filter by Subject", "Search"])
                
                with all_tab:
                    for note in notes[:10]:  # Show most recent 10 notes
                        with st.expander(f"{note['title']} ({note['subject']}) - {note['created_at'].strftime('%Y-%m-%d')}"):
                            # Show a preview of the note
                            st.markdown(note['content'][:500] + "..." if len(note['content']) > 500 else note['content'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"View Full Note", key=f"view_{note['id']}"):
                                    st.session_state['selected_note_id'] = note['id']
                                    st.rerun()
                            
                            with col2:
                                if st.button(f"Delete Note", key=f"delete_{note['id']}"):
                                    if notewriter.delete_note(note['id'], st.session_state['user_id']):
                                        st.success("Note deleted successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete note.")
                
                with subject_tab:
                    # Get unique subjects
                    subjects = sorted(set(note['subject'] for note in notes if note['subject']))
                    
                    selected_subject = st.selectbox("Select Subject", subjects)
                    
                    if selected_subject:
                        subject_notes = [note for note in notes if note['subject'] == selected_subject]
                        
                        for note in subject_notes:
                            with st.expander(f"{note['title']} - {note['created_at'].strftime('%Y-%m-%d')}"):
                                # Show a preview of the note
                                st.markdown(note['content'][:500] + "..." if len(note['content']) > 500 else note['content'])
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"View Full Note", key=f"view_subj_{note['id']}"):
                                        st.session_state['selected_note_id'] = note['id']
                                        st.rerun()
                                
                                with col2:
                                    if st.button(f"Delete Note", key=f"delete_subj_{note['id']}"):
                                        if notewriter.delete_note(note['id'], st.session_state['user_id']):
                                            st.success("Note deleted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to delete note.")
                
                with search_tab:
                    search_query = st.text_input("Search your notes:", placeholder="Enter keywords to search")
                    
                    if search_query:
                        search_results = notewriter.search_notes(st.session_state['user_id'], search_query)
                        
                        if not search_results:
                            st.info(f"No notes found matching '{search_query}'")
                        else:
                            st.write(f"Found {len(search_results)} notes matching '{search_query}'")
                            
                            for note in search_results:
                                with st.expander(f"{note['title']} ({note['subject']}) - {note['created_at'].strftime('%Y-%m-%d')}"):
                                    # Show a preview of the note
                                    st.markdown(note['content'][:500] + "..." if len(note['content']) > 500 else note['content'])
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"View Full Note", key=f"view_search_{note['id']}"):
                                            st.session_state['selected_note_id'] = note['id']
                                            st.rerun()
                                    
                                    with col2:
                                        if st.button(f"Delete Note", key=f"delete_search_{note['id']}"):
                                            if notewriter.delete_note(note['id'], st.session_state['user_id']):
                                                st.success("Note deleted successfully!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to delete note.")
    
    # View selected note in full
    if 'selected_note_id' in st.session_state and 'user_id' in st.session_state:
        notewriter = get_notewriter()
        if notewriter:
            note = notewriter.get_note_by_id(st.session_state['selected_note_id'], st.session_state['user_id'])
            
            if note:
                st.subheader(f"📄 {note['title']}")
                
                # Note metadata
                metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
                with metadata_col1:
                    st.write(f"**Subject:** {note['subject']}")
                with metadata_col2:
                    st.write(f"**Created:** {note['created_at'].strftime('%Y-%m-%d')}")
                with metadata_col3:
                    if note.get('tags'):
                        tags_list = note['tags'] if isinstance(note['tags'], list) else [note['tags']]
                        st.write(f"**Tags:** {', '.join(tags_list)}")
                
                # Tabs for viewing in different formats
                view_tab, mindmap_tab, edit_tab = st.tabs(["View Note", "Mind Map View", "Edit Note"])
                
                with view_tab:
                    st.markdown(note['content'])
                    
                    # Option to clear the selected note
                    if st.button("Back to Notes List", key="back_from_view"):
                        del st.session_state['selected_note_id']
                        st.rerun()
                
                with mindmap_tab:
                    # Check if we already have a mindmap for this note
                    existing_mindmap = notewriter.get_mindmap(note['id'])
                    
                    if existing_mindmap:
                        # Use the existing mindmap
                        st.success("Loading saved mind map")
                        
                        # Display the mindmap
                        st.info("Below is an interactive mind map of your notes. You can expand/collapse branches by clicking on them.")
                        
                        # Use the streamlit-markmap component to visualize
                        from streamlit_markmap import markmap
                        markmap(existing_mindmap, height=600)
                    else:
                        # Generate a new mindmap
                        with st.spinner("Generating Mind Map visualization..."):
                            mindmap_content = notewriter.generate_mindmap(note['content'])
                            # Save the mindmap for future use
                            notewriter.save_mindmap(note['id'], mindmap_content)
                            st.success("Mind map generated and saved for future use")
                            
                            # Display the mindmap
                            st.info("Below is an interactive mind map of your notes. You can expand/collapse branches by clicking on them.")
                            
                            # Use the streamlit-markmap component to visualize
                            from streamlit_markmap import markmap
                            markmap(mindmap_content, height=600)
                
                with edit_tab:
                    # Create a form for editing
                    with st.form("edit_note_form"):
                        edited_title = st.text_input("Title", value=note['title'])
                        edited_subject = st.text_input("Subject", value=note['subject'])
                        edited_content = st.text_area("Content", value=note['content'], height=400)
                        
                        # Convert tags list to string for editing
                        tags_str = ""
                        if note.get('tags'):
                            tags_list = note['tags'] if isinstance(note['tags'], list) else [note['tags']]
                            tags_str = ", ".join(tags_list)
                        
                        edited_tags = st.text_input("Tags (comma separated)", value=tags_str)
                        
                        save_changes = st.form_submit_button("Save Changes")
                        
                        if save_changes:
                            # Update the note
                            update_data = {
                                "title": edited_title,
                                "content": edited_content,
                                "subject": edited_subject,
                                "tags": edited_tags
                            }
                            
                            if notewriter.update_note(note['id'], st.session_state['user_id'], update_data):
                                st.success("Note updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to update note.")
                    
                    # Delete note button (outside the form)
                    if st.button("Delete This Note", key="delete_from_edit"):
                        # Add a confirmation check
                        confirm_delete = st.button("Confirm Delete", key="confirm_delete")
                        
                        if confirm_delete:
                            if notewriter.delete_note(note['id'], st.session_state['user_id']):
                                st.success("Note deleted successfully!")
                                
                                # Clear the selected note from session state
                                del st.session_state['selected_note_id']
                                
                                # Refresh the page
                                st.rerun()
                            else:
                                st.error("Failed to delete note.")

def planner_page():
    st.title("📅 Planner")
    
    st.markdown("""
    The Planner agent helps you manage your academic schedule and optimize your study time.
    """)
    
    # Task management section
    st.subheader("Task Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("task_form"):
            task_title = st.text_input("Task Title")
            task_description = st.text_area("Description", height=100)
            
            task_col1, task_col2 = st.columns(2)
            with task_col1:
                due_date = st.date_input("Due Date")
                due_time = st.time_input("Due Time")
            with task_col2:
                priority = st.selectbox(
                    "Priority",
                    ["High", "Medium", "Low"]
                )
            
            submit_task = st.form_submit_button("Add Task")
            
            if submit_task and task_title:
                if 'user_id' not in st.session_state:
                    st.warning("Please set up your profile first on the Home page.")
                else:
                    # Combine date and time
                    due_datetime = pd.Timestamp.combine(due_date, due_time)
                    
                    # Save task to database
                    conn = init_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO tasks (student_id, title, description, due_date, priority, status)
                        VALUES (%s, %s, %s, %s, %s, 'pending')
                    """, (st.session_state['user_id'], task_title, task_description, due_datetime, priority))
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    st.success(f"Task '{task_title}' added successfully!")
    
    with col2:
        st.subheader("Quick Statistics")
        
        if 'user_id' in st.session_state:
            conn = init_connection()
            cursor = conn.cursor()
            
            # Get task counts by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY status
            """, (st.session_state['user_id'],))
            
            status_counts = dict(cursor.fetchall())
            
            # Get task counts by priority
            cursor.execute("""
                SELECT priority, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY priority
            """, (st.session_state['user_id'],))
            
            priority_counts = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            # Display counts
            pending = status_counts.get('pending', 0)
            completed = status_counts.get('completed', 0)
            
            st.metric("Pending Tasks", pending)
            st.metric("Completed Tasks", completed)
            
            # Display by priority if available
            if priority_counts:
                st.markdown("---")
                st.caption("Tasks by Priority")
                for priority, count in priority_counts.items():
                    st.metric(f"{priority} Priority", count)
        else:
            st.info("Set up your profile to see statistics.")
    
    # Show task list
    st.markdown("---")
    st.subheader("Your Tasks")
    
    if 'user_id' in st.session_state:
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, due_date, priority, status
            FROM tasks
            WHERE student_id = %s
            ORDER BY due_date ASC
        """, (st.session_state['user_id'],))
        
        tasks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if tasks:
            tasks_df = pd.DataFrame(
                tasks, 
                columns=["ID", "Title", "Description", "Due Date", "Priority", "Status"]
            )
            
            tasks_df["Due Date"] = pd.to_datetime(tasks_df["Due Date"]).dt.strftime("%Y-%m-%d %H:%M")
            
            # Display tabs for different task views
            tab1, tab2, tab3 = st.tabs(["All Tasks", "Pending Tasks", "Completed Tasks"])
            
            with tab1:
                display_tasks(tasks_df, tab_id="all")
            
            with tab2:
                display_tasks(tasks_df[tasks_df["Status"] == "pending"], tab_id="pending")
            
            with tab3:
                display_tasks(tasks_df[tasks_df["Status"] == "completed"], tab_id="completed")
        else:
            st.info("You haven't added any tasks yet.")
    else:
        st.warning("Please set up your profile first on the Home page.")

def display_tasks(tasks_df, tab_id="all"):
    """
    Display a list of tasks with action buttons
    
    Args:
        tasks_df: DataFrame of tasks to display
        tab_id: Identifier for the tab (all, pending, completed) to ensure unique keys
    """
    if not tasks_df.empty:
        for index, row in tasks_df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 1, 1, 1])
            
            with col1:
                st.write(f"**{row['Title']}**")
                if row['Description']:
                    st.caption(row['Description'][:50] + "..." if len(row['Description']) > 50 else row['Description'])
            
            with col2:
                st.write(f"Due: {row['Due Date']}")
            
            with col3:
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }
                st.write(f"{priority_color.get(row['Priority'], '⚪')} {row['Priority']}")
            
            with col4:
                status = "✓" if row['Status'] == "completed" else "🕒"
                st.write(f"{status} {row['Status'].capitalize()}")
            
            with col5:
                if row['Status'] == "pending":
                    if st.button("Complete", key=f"complete_task_{tab_id}_{row['ID']}"):
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            UPDATE tasks
                            SET status = 'completed'
                            WHERE id = %s AND student_id = %s
                        """, (row['ID'], st.session_state['user_id']))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        st.rerun()
            
            with col6:
                if st.button("Delete", key=f"delete_task_{tab_id}_{row['ID']}"):
                    st.session_state['delete_task_id'] = row['ID']
                    st.session_state['delete_task_confirmation'] = False
        
        # Show delete confirmation
        if 'delete_task_id' in st.session_state and 'delete_task_confirmation' in st.session_state and not st.session_state['delete_task_confirmation']:
            task_id = st.session_state['delete_task_id']
            st.warning(f"Are you sure you want to delete this task? This action cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", key=f"confirm_delete_task_{tab_id}_{task_id}"):
                    # Import the planner agent
                    planner = get_planner()
                    
                    if planner and planner.delete_task(task_id, st.session_state['user_id']):
                        st.success("Task deleted successfully!")
                        st.session_state['delete_task_confirmation'] = True
                        
                        # Refresh the page
                        st.rerun()
                    else:
                        st.error("Failed to delete task.")
            with col2:
                if st.button("Cancel", key=f"cancel_delete_task_{tab_id}_{task_id}"):
                    # Clear deletion state
                    del st.session_state['delete_task_id']
                    del st.session_state['delete_task_confirmation']
                    st.rerun()
    else:
        st.info("No tasks to display in this category.")

def advisor_page():
    st.title("🧠 Advisor")
    
    st.markdown("""
    The Advisor agent provides personalized learning and time management advice based on your 
    student profile, academic data, and course syllabi.
    """)
    
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for AI advice):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Get user profile
    conn = init_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT learning_style, study_hours
        FROM students
        WHERE id = %s
    """, (st.session_state['user_id'],))
    
    profile = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not profile:
        st.warning("Profile data not found. Please update your profile on the Home page.")
        return
    
    learning_style, study_hours = profile
    
    # Display profile summary
    st.subheader("Your Learning Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Learning Style: **{learning_style}**")
        st.info(f"Daily Study Hours: **{study_hours}**")
    
    with col2:
        learning_style_tips = {
            "Visual": "Use diagrams, charts, and color coding in your notes.",
            "Auditory": "Record lectures and listen to them repeatedly. Discuss topics aloud.",
            "Reading/Writing": "Take detailed notes and rewrite them to reinforce concepts.",
            "Kinesthetic": "Use hands-on activities, experiments, and real-world applications."
        }
        
        st.markdown(f"### Tips for {learning_style} Learners")
        st.markdown(learning_style_tips.get(learning_style, "Customize your study approach."))
    
    # Student Progress Dashboard
    st.markdown("---")
    st.subheader("📊 Academic Progress Dashboard")
    
    # Get user's stats
    conn = init_connection()
    cursor = conn.cursor()
    
    # Notes statistics
    cursor.execute("""
        SELECT COUNT(id), COUNT(DISTINCT subject)
        FROM notes
        WHERE student_id = %s
    """, (st.session_state['user_id'],))
    
    notes_stats = cursor.fetchone()
    total_notes = notes_stats[0] if notes_stats else 0
    unique_subjects = notes_stats[1] if notes_stats else 0
    
    # Task statistics
    cursor.execute("""
        SELECT status, COUNT(id)
        FROM tasks
        WHERE student_id = %s
        GROUP BY status
    """, (st.session_state['user_id'],))
    
    task_stats = dict(cursor.fetchall())
    pending_tasks = task_stats.get('pending', 0)
    completed_tasks = task_stats.get('completed', 0)
    
    # Recent activity - includes notes and chat interactions
    cursor.execute("""
        SELECT 'note' as type, title, created_at 
        FROM notes 
        WHERE student_id = %s
        UNION ALL
        SELECT 'chat' as type, title, created_at
        FROM knowledge_base
        WHERE metadata->>'type' = 'chat_interaction' AND metadata->>'student_id' = %s
        ORDER BY created_at DESC
        LIMIT 5
    """, (st.session_state['user_id'], str(st.session_state['user_id'])))
    
    recent_activity = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Display the stats in a dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Notes", total_notes)
    with col2:
        st.metric("Subjects Covered", unique_subjects)
    with col3:
        st.metric("Pending Tasks", pending_tasks)
    with col4:
        st.metric("Completed Tasks", completed_tasks)
        
    # Display recent activity
    st.subheader("Recent Activity")
    if recent_activity:
        for activity in recent_activity:
            activity_type, title, date = activity
            activity_icon = "📝" if activity_type == "note" else "💬"
            st.write(f"{activity_icon} **{title}** - {date.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No recent activity found. Start creating notes or tasks!")
    
    # Syllabus Upload Section
    st.markdown("---")
    st.subheader("📚 Course Syllabus Analysis")
    
    # Create tabs for Syllabus Upload and Saved Syllabi
    syllabus_tab1, syllabus_tab2 = st.tabs(["Upload Syllabus", "Saved Syllabi"])
    
    with syllabus_tab1:
        st.markdown("""
        Upload your course syllabus to get personalized advice tailored to your specific courses.
        This helps the advisor understand your course requirements, deadlines, and topics.
        """)
        
        uploaded_syllabus = st.file_uploader("Upload Course Syllabus (PDF)", type=["pdf"])
        
        if uploaded_syllabus is not None:
            # Form for syllabus metadata
            with st.form("syllabus_form"):
                course_name = st.text_input("Course Name", placeholder="e.g., Introduction to Computer Science")
                course_code = st.text_input("Course Code", placeholder="e.g., CS101")
                semester = st.text_input("Semester", placeholder="e.g., Fall 2025")
                
                save_syllabus = st.form_submit_button("Save Syllabus")
                
                if save_syllabus and course_name and course_code:
                    try:
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_syllabus.getvalue())
                            temp_path = temp_file.name
                        
                        # Load PDF using LangChain
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()
                        
                        # Extract text content
                        syllabus_content = "\n".join([doc.page_content for doc in documents])
                        
                        # Clean up the temporary file
                        os.unlink(temp_path)
                        
                        # Save to database
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s) 
                            RETURNING id
                        """, (
                            f"Syllabus: {course_name} ({course_code})",
                            syllabus_content,
                            json.dumps({
                                "type": "syllabus",
                                "course_name": course_name,
                                "course_code": course_code,
                                "semester": semester,
                                "student_id": st.session_state['user_id']
                            })
                        ))
                        
                        syllabus_id = cursor.fetchone()[0]
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        st.success(f"Syllabus for {course_name} ({course_code}) saved successfully!")
                        
                        # Set the current syllabus for advice
                        st.session_state['current_syllabus_id'] = syllabus_id
                        st.session_state['current_syllabus_name'] = f"{course_name} ({course_code})"
                        
                    except Exception as e:
                        st.error(f"Error processing syllabus: {str(e)}")
    
    with syllabus_tab2:
        # Display saved syllabi
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, created_at, metadata
            FROM knowledge_base
            WHERE 
                metadata->>'type' = 'syllabus'
                AND metadata->>'student_id' = %s
            ORDER BY created_at DESC
        """, (str(st.session_state['user_id']),))
        
        syllabi = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if syllabi:
            st.markdown("### Your Saved Syllabi")
            
            for syllabus in syllabi:
                syllabus_id, title, created_at, metadata = syllabus
                # Handle metadata - check if it's already a dictionary
                if isinstance(metadata, dict):
                    metadata_dict = metadata
                else:
                    metadata_dict = json.loads(metadata)
                
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{title}**")
                    st.caption(f"Semester: {metadata_dict.get('semester', 'N/A')}")
                
                with col2:
                    created_date = pd.to_datetime(created_at).strftime("%Y-%m-%d")
                    st.write(f"Added: {created_date}")
                
                with col3:
                    if st.button("Select", key=f"select_syllabus_{syllabus_id}"):
                        st.session_state['current_syllabus_id'] = syllabus_id
                        st.session_state['current_syllabus_name'] = title
                        st.success(f"Selected {title} for advice")
        else:
            st.info("You haven't uploaded any syllabi yet. Upload syllabi to get course-specific advice.")
    
    # AI Advisor Section
    st.markdown("---")
    st.subheader("Ask the Advisor")
    
    # Display current selected syllabus if any
    if 'current_syllabus_id' in st.session_state:
        st.info(f"Current syllabus: **{st.session_state.get('current_syllabus_name', 'None')}**")
    
    user_question = st.text_area("What would you like advice on?", 
                              placeholder="e.g., How can I prepare for my upcoming exam? OR What topics should I focus on for CS101?")
    
    if st.button("Get Comprehensive Advice") and user_question:
        if not api_key:
            st.warning("Please enter your Groq API key to enable AI advice.")
        else:
            # Gather comprehensive student data for context
            conn = init_connection()
            cursor = conn.cursor()
            
            # Get task count by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY status
            """, (st.session_state['user_id'],))
            
            task_stats = dict(cursor.fetchall())
            
            # Get note count by subject
            cursor.execute("""
                SELECT subject, COUNT(*) 
                FROM notes 
                WHERE student_id = %s AND subject != ''
                GROUP BY subject
            """, (st.session_state['user_id'],))
            
            subject_stats = dict(cursor.fetchall())
            
            # Get upcoming deadlines
            cursor.execute("""
                SELECT title, due_date, priority
                FROM tasks
                WHERE student_id = %s AND status = 'pending' AND due_date >= NOW()
                ORDER BY due_date ASC
                LIMIT 5
            """, (st.session_state['user_id'],))
            
            upcoming_tasks = cursor.fetchall()
            
            # Get recent notes
            cursor.execute("""
                SELECT title, subject, created_at
                FROM notes
                WHERE student_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """, (st.session_state['user_id'],))
            
            recent_notes = cursor.fetchall()
            
            # Get current syllabus content if selected
            syllabus_content = ""
            if 'current_syllabus_id' in st.session_state:
                cursor.execute("""
                    SELECT content, metadata
                    FROM knowledge_base
                    WHERE id = %s
                """, (st.session_state['current_syllabus_id'],))
                
                syllabus_result = cursor.fetchone()
                if syllabus_result:
                    syllabus_content = syllabus_result[0]
                    syllabus_metadata_raw = syllabus_result[1]
                    
                    # Handle metadata - check if it's already a dictionary
                    if isinstance(syllabus_metadata_raw, dict):
                        syllabus_metadata = syllabus_metadata_raw
                    else:
                        syllabus_metadata = json.loads(syllabus_metadata_raw)
            
            # Get chat interactions for learning patterns
            cursor.execute("""
                SELECT content
                FROM knowledge_base
                WHERE metadata->>'type' = 'chat_interaction' 
                AND metadata->>'student_id' = %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (str(st.session_state['user_id']),))
            
            chat_interactions = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            with st.spinner("Generating comprehensive academic advice..."):
                try:
                    # Initialize LLM
                    llm = GroqLLaMa(api_key)
                    
                    # Create a comprehensive context-rich prompt
                    prompt = f"""
                    As an academic advisor, provide comprehensive personalized advice for a student with the following profile:
                    
                    STUDENT PROFILE:
                    Learning Style: {learning_style}
                    Daily Study Hours: {study_hours}
                    
                    ACADEMIC STATS:
                    - Total Notes: {total_notes}
                    - Subjects Covered: {unique_subjects}
                    - Completed tasks: {task_stats.get('completed', 0)}
                    - Pending tasks: {task_stats.get('pending', 0)}
                    
                    SUBJECT DISTRIBUTION:
                    {', '.join([f"{subject}: {count}" for subject, count in subject_stats.items()])}
                    
                    UPCOMING DEADLINES:
                    """
                    
                    # Add upcoming tasks if available
                    if upcoming_tasks:
                        for task in upcoming_tasks:
                            task_title, due_date, priority = task
                            prompt += f"- {task_title} (Due: {due_date.strftime('%Y-%m-%d')}, Priority: {priority})\n"
                    else:
                        prompt += "No upcoming deadlines.\n"
                    
                    prompt += "\nRECENT NOTES:\n"
                    
                    # Add recent notes if available
                    if recent_notes:
                        for note in recent_notes:
                            note_title, note_subject, created_at = note
                            prompt += f"- {note_title} ({note_subject}, Created: {created_at.strftime('%Y-%m-%d')})\n"
                    else:
                        prompt += "No recent notes.\n"
                    
                    # Add chat interactions for learning patterns if available
                    if chat_interactions:
                        prompt += "\nRECENT LEARNING INTERACTIONS:\n"
                        for i, chat in enumerate(chat_interactions[:5]):  # Limit to 5 to manage token count
                            # Handle chat data - check if it's already a dictionary
                            chat_content = chat[0]
                            if isinstance(chat_content, dict):
                                chat_data = chat_content
                            else:
                                chat_data = json.loads(chat_content)
                            prompt += f"- Question: {chat_data.get('question', 'N/A')}\n"
                    
                    # Add syllabus context if available
                    if syllabus_content:
                        # Truncate syllabus content if too long (to avoid token limits)
                        max_syllabus_length = 2000
                        truncated_syllabus = syllabus_content[:max_syllabus_length]
                        if len(syllabus_content) > max_syllabus_length:
                            truncated_syllabus += "... [content truncated]"
                        
                        prompt += f"""
                        
                        COURSE SYLLABUS INFORMATION:
                        
                        Course: {syllabus_metadata.get('course_name')} ({syllabus_metadata.get('course_code')})
                        Semester: {syllabus_metadata.get('semester')}
                        
                        Syllabus Content:
                        {truncated_syllabus}
                        """
                    
                    prompt += f"""
                    
                    The student is asking: {user_question}
                    
                    PROVIDE COMPREHENSIVE ADVICE THAT:
                    1. Analyzes their current academic situation holistically
                    2. Takes into account their learning style, subject distribution, and time commitments
                    3. Provides specific strategies tailored to their situation
                    4. Includes actionable steps they can take immediately
                    5. Addresses any upcoming deadlines or course requirements
                    6. Recommends specific study techniques based on their learning style
                    7. Suggests how to optimize their study time based on their available hours
                    
                    If the student has provided a course syllabus, analyze it to provide advice specific to:
                    - Important topics and concepts in the course
                    - Upcoming assignments or exams mentioned in the syllabus
                    - Recommended study techniques for the specific course material
                    - How to budget time effectively for this course
                    
                    Your advice should be comprehensive, specific, and actionable.
                    """
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Generate advice
                    advice = llm.generate(messages)
                    
                    # Display advice
                    st.markdown("### 💡 Comprehensive Academic Advice")
                    st.markdown(advice)
                    
                    # Save the advice to knowledge base for future reference
                    try:
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s)
                        """, (
                            f"Advice: {user_question[:50]}{'...' if len(user_question) > 50 else ''}",
                            advice,
                            json.dumps({
                                "type": "advisor_advice",
                                "question": user_question,
                                "student_id": st.session_state['user_id'],
                                "timestamp": datetime.now().isoformat(),
                                "syllabus_id": st.session_state.get('current_syllabus_id')
                            })
                        ))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        print(f"Error storing advice: {str(db_error)}")
                    
                except Exception as e:
                    st.error(f"Error generating advice: {str(e)}")
                    st.warning("Please check your API key and try again.")
    
    # Show advice history
    st.markdown("---")
    st.subheader("Previous Advice")
    
    conn = init_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, content, created_at
        FROM knowledge_base
        WHERE 
            metadata->>'type' = 'advisor_advice'
            AND metadata->>'student_id' = %s
        ORDER BY created_at DESC
        LIMIT 5
    """, (str(st.session_state['user_id']),))
    
    previous_advice = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if previous_advice:
        for i, advice in enumerate(previous_advice):
            title, content, date = advice
            with st.expander(f"{title} - {date.strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(content)
    else:
        st.info("No previous advice found. Ask the advisor for advice to get started.")

def pdf_chat_page():
    st.title("💬 PDF & Notes Chat")
    
    st.markdown("""
    Chat with your study materials. Ask questions about your saved notes or upload a PDF to get instant answers.
    This feature uses RAG (Retrieval Augmented Generation) technology to provide precise answers from your documents.
    """)
    
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for chat):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Create tabs for different content sources
    source_tab1, source_tab2, source_tab3 = st.tabs(["📄 PDF Upload", "📝 Saved Notes", "🔍 Multi-Source Search"])
    
    # For tracking the current content source
    if 'chat_content' not in st.session_state:
        st.session_state.chat_content = None
    
    if 'chat_content_name' not in st.session_state:
        st.session_state.chat_content_name = None
        
    # For storing RAG pipelines to avoid rebuilding them
    if 'rag_pipelines' not in st.session_state:
        st.session_state.rag_pipelines = {}
        
    # For storing multi-source selections
    if 'multi_sources' not in st.session_state:
        st.session_state.multi_sources = []
    
    with source_tab1:
        st.subheader("Upload a PDF to Chat")
        
        uploaded_pdf = st.file_uploader("Upload PDF document:", type=["pdf"])
        
        if uploaded_pdf is not None:
            # Extract PDF content
            try:
                # Display processing message
                with st.spinner("Processing PDF and building knowledge base..."):
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_pdf.getvalue())
                        temp_path = temp_file.name
                    
                    # Load PDF using LangChain - this creates Document objects with metadata
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()
                    
                    # Clean up the temporary file
                    os.unlink(temp_path)
                    
                    # Extract text content for preview
                    pdf_content = "\n".join([doc.page_content for doc in documents])
                    
                    # Store in session state
                    pdf_name = uploaded_pdf.name
                    st.session_state.chat_content = documents  # Store document objects
                    st.session_state.chat_content_name = pdf_name
                    
                    # Clear any existing RAG pipeline for this document to rebuild it
                    pipeline_key = f"rag_{hash(pdf_name)}"
                    if pipeline_key in st.session_state.rag_pipelines:
                        del st.session_state.rag_pipelines[pipeline_key]
                    
                    st.success(f"Loaded and indexed PDF: {pdf_name}")
                    
                    # Display a preview of the content
                    with st.expander("PDF Content Preview"):
                        st.text(pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    with source_tab2:
        st.subheader("Chat with Your Notes")
        
        # Get user's notes
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, subject, created_at 
            FROM notes 
            WHERE student_id = %s
            ORDER BY created_at DESC
        """, (st.session_state['user_id'],))
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if notes:
            notes_options = ["Select a note..."] + [f"{note[1]} ({note[2]})" for note in notes]
            selected_note_index = st.selectbox("Choose a note to chat with:", 
                                              options=range(len(notes_options)),
                                              format_func=lambda x: notes_options[x])
            
            if selected_note_index > 0:
                # Get the actual note
                note_id = notes[selected_note_index - 1][0]
                
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, content
                    FROM notes
                    WHERE id = %s AND student_id = %s
                """, (note_id, st.session_state['user_id']))
                
                note = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if note:
                    # Display processing message
                    with st.spinner("Processing note and building knowledge base..."):
                        note_title, note_content = note
                        
                        # Store in session state
                        st.session_state.chat_content = note_content
                        st.session_state.chat_content_name = note_title
                        
                        # Clear any existing RAG pipeline for this note to rebuild it
                        pipeline_key = f"rag_{hash(note_title)}"
                        if pipeline_key in st.session_state.rag_pipelines:
                            del st.session_state.rag_pipelines[pipeline_key]
                        
                        st.success(f"Loaded and indexed note: {note_title}")
                        
                        # Display a preview of the content
                        with st.expander("Note Content Preview"):
                            st.text(note_content[:500] + "..." if len(note_content) > 500 else note_content)
        else:
            st.info("You haven't saved any notes yet. Create notes using the Notewriter feature.")
    
    with source_tab3:
        st.subheader("Multi-Source Knowledge Base")
        st.markdown("""
        Select multiple notes and PDFs to build a comprehensive knowledge base. 
        This allows you to ask questions across all your study materials at once.
        """)
        
        # Get user's notes for selection
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, subject
            FROM notes 
            WHERE student_id = %s
            ORDER BY subject, title
        """, (st.session_state['user_id'],))
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Get syllabi from knowledge base
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, metadata
            FROM knowledge_base
            WHERE 
                metadata->>'type' = 'syllabus'
                AND metadata->>'student_id' = %s
            ORDER BY title
        """, (str(st.session_state['user_id']),))
        
        syllabi = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Display available sources with checkboxes
        st.subheader("Select Sources")
        
        with st.form("multi_source_form"):
            st.markdown("### Notes")
            selected_notes = []
            if notes:
                for note_id, title, subject in notes:
                    if st.checkbox(f"{title} ({subject})", key=f"note_{note_id}"):
                        selected_notes.append(note_id)
            else:
                st.info("No notes available. Create notes using the Notewriter feature.")
                
            st.markdown("### Syllabi")
            selected_syllabi = []
            if syllabi:
                for syllabus_id, title, _ in syllabi:
                    if st.checkbox(f"{title}", key=f"syllabus_{syllabus_id}"):
                        selected_syllabi.append(syllabus_id)
            else:
                st.info("No syllabi available. Upload syllabi in the Advisor section.")
                
            build_kb_button = st.form_submit_button("Build Multi-Source Knowledge Base")
            
            if build_kb_button:
                if not selected_notes and not selected_syllabi:
                    st.warning("Please select at least one source to build the knowledge base.")
                else:
                    with st.spinner("Building multi-source knowledge base..."):
                        # Get content for all selected sources
                        sources_list = []
                        
                        # Get selected notes
                        if selected_notes:
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            placeholders = ', '.join(['%s'] * len(selected_notes))
                            query = f"""
                                SELECT id, title, content
                                FROM notes
                                WHERE id IN ({placeholders}) AND student_id = %s
                            """
                            
                            cursor.execute(query, selected_notes + [st.session_state['user_id']])
                            note_data = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            for note_id, title, content in note_data:
                                sources_list.append((content, f"Note: {title}"))
                        
                        # Get selected syllabi
                        if selected_syllabi:
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            placeholders = ', '.join(['%s'] * len(selected_syllabi))
                            query = f"""
                                SELECT id, title, content
                                FROM knowledge_base
                                WHERE id IN ({placeholders})
                            """
                            
                            cursor.execute(query, selected_syllabi)
                            syllabus_data = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            for syllabus_id, title, content in syllabus_data:
                                sources_list.append((content, f"Syllabus: {title}"))
                        
                        # Combine all sources
                        all_documents = combine_knowledge_sources(sources_list)
                        
                        # Store for chat interface
                        st.session_state.chat_content = all_documents
                        st.session_state.chat_content_name = f"Multi-Source ({len(sources_list)} documents)"
                        
                        # Clear any existing RAG pipeline
                        pipeline_key = f"rag_multi_source_{hash(tuple(selected_notes + selected_syllabi))}"
                        if pipeline_key in st.session_state.rag_pipelines:
                            del st.session_state.rag_pipelines[pipeline_key]
                            
                        # Pre-initialize pipeline if API key is available
                        if api_key:
                            with st.spinner("Building knowledge retrieval system..."):
                                llm = GroqLLaMa(api_key)
                                qa_chain = create_rag_pipeline(
                                    all_documents,
                                    f"Multi-Source ({len(sources_list)} documents)",
                                    llm
                                )
                                st.session_state.rag_pipelines[pipeline_key] = qa_chain
                        
                        st.success(f"Built knowledge base with {len(sources_list)} sources. You can now chat with all selected materials!")
        
    # Chat interface
    st.markdown("---")
    
    if st.session_state.chat_content:
        st.subheader(f"Chat with: {st.session_state.chat_content_name}")
        
        # Initialize chat history for this content if not exists
        chat_history_key = f"chat_history_{hash(st.session_state.chat_content_name)}"
        if chat_history_key not in st.session_state:
            # Check if we have history in the database first
            try:
                conn = init_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content
                    FROM knowledge_base
                    WHERE 
                        metadata->>'type' = 'chat_history'
                        AND metadata->>'content_name_hash' = %s
                        AND metadata->>'student_id' = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (str(hash(st.session_state.chat_content_name)), str(st.session_state['user_id'])))
                
                saved_history = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if saved_history and saved_history[0]:
                    try:
                        # Load chat history from database
                        st.session_state[chat_history_key] = json.loads(saved_history[0])
                    except:
                        # If loading fails, start with empty history
                        st.session_state[chat_history_key] = []
                else:
                    # No history found in database
                    st.session_state[chat_history_key] = []
            except:
                # If database query fails, start with empty history
                st.session_state[chat_history_key] = []
        
        # Display chat history
        with st.container():
            # Create a scrollable container for chat history
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state[chat_history_key]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        
                        # Display sources if available
                        if "sources" in message and message["sources"]:
                            st.caption("**Sources:**")
                            for source in message["sources"]:
                                st.caption(f"- {source}")
        
        # Chat input
        user_question = st.chat_input("Ask a question about this content...")
        
        if user_question and api_key:
            # Add user message to chat history
            st.session_state[chat_history_key].append({
                "role": "user", 
                "content": user_question,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # Initialize LLM
                    llm = GroqLLaMa(api_key)
                    
                    # Get or create RAG pipeline for this content
                    pipeline_key = f"rag_{hash(st.session_state.chat_content_name)}"
                    
                    if pipeline_key not in st.session_state.rag_pipelines:
                        with st.spinner("Building knowledge retrieval system..."):
                            # Create a new RAG pipeline
                            qa_chain = create_rag_pipeline(
                                st.session_state.chat_content, 
                                st.session_state.chat_content_name,
                                llm
                            )
                            # Cache the pipeline for future use
                            st.session_state.rag_pipelines[pipeline_key] = qa_chain
                    else:
                        # Use existing pipeline
                        qa_chain = st.session_state.rag_pipelines[pipeline_key]
                    
                    # Execute the query against the RAG pipeline
                    with st.spinner("Searching document and generating answer..."):
                        result = qa_chain({"query": user_question})
                        
                        # Extract the response and source documents
                        response = result["result"]
                        source_docs = result["source_documents"]
                        
                        # Extract source information
                        sources = []
                        for doc in source_docs:
                            source_info = []
                            if "source" in doc.metadata:
                                source_info.append(doc.metadata["source"])
                            if "chunk_id" in doc.metadata:
                                source_info.append(f"Chunk {doc.metadata['chunk_id']}")
                            sources.append(" - ".join(source_info))
                    
                    # Add assistant response to chat history with sources
                    st.session_state[chat_history_key].append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Save chat history to database for persistence
                    try:
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        # First, store the interaction details
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s)
                        """, (
                            f"Chat: {st.session_state.chat_content_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            json.dumps({
                                "question": user_question,
                                "answer": response,
                                "sources": sources
                            }),
                            json.dumps({
                                "type": "chat_interaction",
                                "content_name": st.session_state.chat_content_name,
                                "student_id": st.session_state['user_id'],
                                "timestamp": datetime.now().isoformat()
                            })
                        ))
                        
                        # Then, update the full chat history
                        # First check if we have an existing history record
                        cursor.execute("""
                            SELECT id FROM knowledge_base
                            WHERE 
                                metadata->>'type' = 'chat_history'
                                AND metadata->>'content_name_hash' = %s
                                AND metadata->>'student_id' = %s
                        """, (str(hash(st.session_state.chat_content_name)), str(st.session_state['user_id'])))
                        
                        existing_history = cursor.fetchone()
                        
                        if existing_history:
                            # Update existing record
                            cursor.execute("""
                                UPDATE knowledge_base
                                SET content = %s, metadata = %s
                                WHERE id = %s
                            """, (
                                json.dumps(st.session_state[chat_history_key]),
                                json.dumps({
                                    "type": "chat_history",
                                    "content_name": st.session_state.chat_content_name,
                                    "content_name_hash": str(hash(st.session_state.chat_content_name)),
                                    "student_id": st.session_state['user_id'],
                                    "updated_at": datetime.now().isoformat()
                                }),
                                existing_history[0]
                            ))
                        else:
                            # Insert new record
                            cursor.execute("""
                                INSERT INTO knowledge_base (title, content, metadata)
                                VALUES (%s, %s, %s)
                            """, (
                                f"Chat History: {st.session_state.chat_content_name}",
                                json.dumps(st.session_state[chat_history_key]),
                                json.dumps({
                                    "type": "chat_history",
                                    "content_name": st.session_state.chat_content_name,
                                    "content_name_hash": str(hash(st.session_state.chat_content_name)),
                                    "student_id": st.session_state['user_id'],
                                    "updated_at": datetime.now().isoformat()
                                })
                            ))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        print(f"Error storing chat history: {str(db_error)}")
                    
                    # Update the placeholder with the response
                    # First, clear the placeholder to remove the "Thinking..." message
                    message_placeholder.empty()
                    
                    # Then create content within the assistant message block
                    st.markdown(response)
                    
                    # Display sources if available
                    if sources:
                        st.caption("**Sources:**")
                        for source in sources:
                            st.caption(f"- {source}")
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    # Clear the placeholder and display the error
                    message_placeholder.empty()
                    st.markdown(error_msg)
                    
                    # Add error to chat history
                    st.session_state[chat_history_key].append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
        elif not api_key and user_question:
            st.warning("Please enter your Groq API key to enable chat.")
    else:
        st.info("Upload a PDF, select a saved note, or build a multi-source knowledge base to start chatting.")
        
    # Add option to clear chat history
    if st.session_state.chat_content and chat_history_key in st.session_state and st.session_state[chat_history_key]:
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            if st.button("Clear Chat History"):
                # Clear from session state
                st.session_state[chat_history_key] = []
                
                # Clear from database
                try:
                    conn = init_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM knowledge_base
                        WHERE 
                            metadata->>'type' = 'chat_history'
                            AND metadata->>'content_name_hash' = %s
                            AND metadata->>'student_id' = %s
                    """, (str(hash(st.session_state.chat_content_name)), str(st.session_state['user_id'])))
                    conn.commit()
                    cursor.close()
                    conn.close()
                except Exception as e:
                    print(f"Error clearing chat history from database: {str(e)}")
                
                st.success("Chat history cleared.")
                st.rerun()
        with col2:
            if st.button("Clear Knowledge Base (Reset Index)"):
                # Clear the RAG pipeline to force rebuilding it
                pipeline_key = f"rag_{hash(st.session_state.chat_content_name)}"
                if pipeline_key in st.session_state.rag_pipelines:
                    del st.session_state.rag_pipelines[pipeline_key]
                st.success("Knowledge base cleared and will be rebuilt on your next question.")
                st.rerun()
        with col3:
            if st.button("Export Chat History"):
                # Export chat history as a note
                try:
                    chat_content = f"# Chat with {st.session_state.chat_content_name}\n\n"
                    chat_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    
                    for message in st.session_state[chat_history_key]:
                        role = "You" if message["role"] == "user" else "Assistant"
                        chat_content += f"## {role}:\n{message['content']}\n\n"
                        if message["role"] == "assistant" and "sources" in message and message["sources"]:
                            chat_content += "**Sources:**\n"
                            for source in message["sources"]:
                                chat_content += f"- {source}\n"
                            chat_content += "\n"
                    
                    conn = init_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO notes (
                            student_id, title, content, subject, source_type
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        st.session_state['user_id'],
                        f"Chat History: {st.session_state.chat_content_name}",
                        chat_content,
                        "Chat History",
                        "chat_export"
                    ))
                    
                    note_id = cursor.fetchone()[0]
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    st.success(f"Chat history exported as a note. You can access it in the Notewriter section.")
                    
                except Exception as e:
                    st.error(f"Error exporting chat history: {str(e)}")

def create_rag_pipeline(content, content_name, llm):
    """
    Create a RAG (Retrieval Augmented Generation) pipeline for document Q&A.
    
    Args:
        content (str): The document content to process
        content_name (str): Name of the document for metadata
        llm: The language model to use for generation
        
    Returns:
        A retrieval QA chain for answering questions about the document
    """
    # Create documents for ingestion
    if isinstance(content, str):
        # Create a Document object with metadata
        document = Document(
            page_content=content,
            metadata={"source": content_name}
        )
        documents = [document]
    else:
        # If it's already a list of Documents
        documents = content
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Add chunk IDs to document metadata
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
    
    # Initialize embeddings - use a lightweight model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store for similarity search
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 chunks for each query
    )
    
    # Create prompt template
    prompt_template = """
    You are an academic assistant helping a student understand their study materials.
    Use only the following retrieved context to answer the question. If you don't know the 
    answer or if it's not in the context, say that you don't have that information in the material.
    
    Be accurate, helpful, clear, and concise.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # If we have a custom GroqLLaMa instance, use its chat_model attribute
    if hasattr(llm, 'chat_model'):
        llm_for_chain = llm.chat_model
    else:
        llm_for_chain = llm
    
    # Create chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_chain,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def combine_knowledge_sources(sources_list):
    """
    Combine multiple knowledge sources (notes, PDFs) into a single collection
    for unified querying.
    
    Args:
        sources_list: List of tuples (content, name) to combine
        
    Returns:
        A RAG pipeline that can query across all sources
    """
    # Create a list of documents
    all_documents = []
    
    for content, name in sources_list:
        # Handle different content types
        if isinstance(content, list) and all(isinstance(item, Document) for item in content):
            # It's already a list of Document objects
            # Add source name to metadata
            for doc in content:
                doc.metadata["source"] = name
            all_documents.extend(content)
        else:
            # It's raw text, create a Document
            document = Document(
                page_content=content,
                metadata={"source": name}
            )
            all_documents.append(document)
    
    return all_documents

def quiz_analyze_page():
    st.title("📚 Quiz & Analyze")
    
    st.markdown("""
    Generate multiple-choice questions from your notes, PDFs, or any content you're studying.
    Test your knowledge and get AI-powered analysis of your performance with actionable feedback.
    """)
    
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Get API key from .env or session state
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key or groq_api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            groq_api_key = st.session_state.api_key
        else:
            groq_api_key = st.text_input("Enter Groq API Key (required for AI processing):", type="password")
            if groq_api_key:
                st.session_state.api_key = groq_api_key
    
    # Create tabs for different quiz functions
    tab1, tab2 = st.tabs(["Create Quiz", "Quiz History"])
    
    with tab1:
        st.subheader("Create a New Quiz")
        
        # Content source selection
        st.markdown("### Step 1: Select Content Source")
        
        source_options = ["Upload PDF", "Enter Text", "From Saved Notes"]
        source_type = st.radio("Choose your content source:", source_options, horizontal=True)
        
        # Content input based on source type
        content = None
        content_source = None
        
        if source_type == "Upload PDF":
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded_file:
                try:
                    with st.spinner("Processing PDF..."):
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        
                        # Extract content using PyPDFLoader
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()
                        
                        # Clean up the temporary file
                        os.unlink(temp_path)
                        
                        # Extract text content
                        content = "\n".join([doc.page_content for doc in documents])
                        content_source = uploaded_file.name
                        
                        # Show preview of the content
                        with st.expander("Content Preview"):
                            st.text(content[:500] + "..." if len(content) > 500 else content)
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        elif source_type == "Enter Text":
            content = st.text_area("Enter or paste your content here:", height=200)
            content_source = "Manual Input"
            if content:
                with st.expander("Content Preview"):
                    st.text(content[:500] + "..." if len(content) > 500 else content)
        
        elif source_type == "From Saved Notes":
            # Get user's notes
            conn = init_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, subject, created_at 
                FROM notes 
                WHERE student_id = %s
                ORDER BY created_at DESC
            """, (st.session_state['user_id'],))
            
            notes = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if notes:
                notes_options = ["Select a note..."] + [f"{note[1]} ({note[2]})" for note in notes]
                selected_note_index = st.selectbox("Choose a note to create quiz from:", 
                                                  options=range(len(notes_options)),
                                                  format_func=lambda x: notes_options[x])
                
                if selected_note_index > 0:
                    # Get the actual note
                    note_id = notes[selected_note_index - 1][0]
                    
                    conn = init_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT title, content, subject
                        FROM notes
                        WHERE id = %s AND student_id = %s
                    """, (note_id, st.session_state['user_id']))
                    
                    note = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    
                    if note:
                        note_title, note_content, subject = note
                        content = note_content
                        content_source = note_title
                        
                        # Show preview of the content
                        with st.expander("Content Preview"):
                            st.text(content[:500] + "..." if len(content) > 500 else content)
            else:
                st.info("You haven't saved any notes yet. Create notes using the Notewriter feature.")
        
        # Quiz settings
        st.markdown("### Step 2: Quiz Settings")
        col1, col2 = st.columns(2)
        with col1:
            quiz_title = st.text_input("Quiz Title", placeholder="e.g., Chapter 5 Review")
            subject = st.text_input("Subject", placeholder="e.g., Biology")
        with col2:
            difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
            num_questions = st.number_input("Number of Questions", min_value=3, max_value=20, value=5)
        
        # Quiz generation
        if st.button("Generate Quiz") and content and groq_api_key:
            if not content or len(content.strip()) < 100:
                st.error("Please provide more content to generate meaningful questions (minimum 100 characters).")
            elif not quiz_title:
                st.error("Please provide a title for your quiz.")
            else:
                with st.spinner(f"Generating {num_questions} {difficulty.lower()}-level questions..."):
                    try:
                        # Initialize model
                        model = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
                        
                        # Limit content size to avoid token limits
                        max_content_length = 25000
                        if len(content) > max_content_length:
                            content = content[:max_content_length]
                        
                        # Create prompt template
                        prompt = ChatPromptTemplate.from_template(
                            """You're an expert quiz creator specializing in {difficulty} level questions. 
                            Generate {num_questions} high-quality MCQs based EXCLUSIVELY on the following content:
                            {content}
                            
                            Requirements:
                            - Each question must cover different key concepts
                            - Questions should progress from basic to advanced (for higher difficulty)
                            - Format each question as:
                                Q1. [Question text]
                                a) [Option A]
                                b) [Option B]
                                c) [Option C]
                                d) [Option D]
                            - Provide answer key in format:
                                Answer Key:
                                1. [correct_letter]
                                2. [correct_letter]
                                ...
                                {num_questions}. [correct_letter]
                            - Avoid markdown formatting
                            - Ensure answers are factually correct based on provided content"""
                        )
                        
                        # Create chain and run it
                        chain = prompt | model
                        response = chain.invoke({
                            "difficulty": difficulty, 
                            "content": content,
                            "num_questions": num_questions
                        })
                        
                        mcq_response = response.content
                        
                        # Parse the response into questions and answers
                        try:
                            # Split questions and answers
                            parts = mcq_response.split("Answer Key:")
                            question_part = parts[0].strip() if len(parts) > 0 else ""
                            answer_part = parts[1].strip() if len(parts) > 1 else ""
                            
                            # Extract questions using regex
                            question_blocks = re.split(r'(?:^|\n)(?:Q?(\d+)\.)', question_part)
                            
                            # Process question blocks
                            questions = []
                            current_q = None
                            for block in question_blocks:
                                if not block:
                                    continue
                                if block.isdigit():
                                    current_q = int(block)
                                elif current_q is not None:
                                    # Add the question with its number
                                    questions.append(f"Q{current_q}. {block.strip()}")
                                    current_q = None
                            
                            # Extract answers
                            answer_entries = re.findall(r'(?:^|\n)(?:Q?(\d+)\.?\s*([a-d]))', answer_part, re.IGNORECASE)
                            answer_dict = {int(num): letter.lower() for num, letter in answer_entries}
                            
                            # Create ordered answer key
                            answer_key = []
                            for i in range(1, num_questions + 1):
                                if i in answer_dict:
                                    answer_key.append(answer_dict[i])
                                else:
                                    # If missing answers, add placeholder
                                    answer_key.append("")
                            
                            # Validate results
                            if len(questions) < 1 or len(answer_key) < 1:
                                st.error(f"Failed to parse quiz content. Please try again.")
                            else:
                                # Store in session state
                                st.session_state.quiz_questions = questions[:num_questions]
                                st.session_state.quiz_answer_key = answer_key[:num_questions]
                                st.session_state.quiz_user_answers = [""] * len(st.session_state.quiz_questions)
                                st.session_state.quiz_content = content
                                st.session_state.quiz_title = quiz_title
                                st.session_state.quiz_subject = subject
                                st.session_state.quiz_difficulty = difficulty
                                st.session_state.quiz_content_source = content_source
                                
                                st.success(f"Quiz generated successfully with {len(st.session_state.quiz_questions)} questions!")
                                st.experimental_rerun()  # Force a rerun to show the quiz
                        except Exception as e:
                            st.error(f"Error parsing quiz: {str(e)}")
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
        
        # Display quiz if questions have been generated
        if 'quiz_questions' in st.session_state and st.session_state.quiz_questions:
            st.markdown("---")
            st.subheader(f"Quiz: {st.session_state.quiz_title}")
            st.markdown(f"**Subject:** {st.session_state.quiz_subject} | **Difficulty:** {st.session_state.quiz_difficulty}")
            
            # Display questions with radio buttons for answers
            for i, question in enumerate(st.session_state.quiz_questions):
                q_text = question.split("a)")[0].strip()
                st.markdown(f"**{i+1}. {q_text}**")
                
                # Extract options using improved parsing
                options = []
                option_labels = {}
                
                # More robust option extraction that handles various formatting
                option_pattern = re.compile(r'([a-d])\)(.*?)(?=\s*[a-d]\)|$)', re.DOTALL)
                
                # Find all options in the question text
                full_question_text = question
                matches = option_pattern.findall(full_question_text)
                
                for opt_letter, opt_text in matches:
                    opt_letter = opt_letter.lower()
                    opt_text = opt_text.strip()
                    option_labels[opt_letter] = opt_text
                    options.append(opt_letter)
                
                # If no options were found with the regex, fall back to the old method
                if len(options) == 0:
                    if "a)" in question:
                        options_text = question.split("a)")[1]
                        options_lines = options_text.split("\n")
                        
                        for line in options_lines:
                            line = line.strip()
                            if line.startswith("a)"):
                                option_labels["a"] = line[2:].strip()
                                options.append("a")
                            elif line.startswith("b)"):
                                option_labels["b"] = line[2:].strip()
                                options.append("b")
                            elif line.startswith("c)"):
                                option_labels["c"] = line[2:].strip()
                                options.append("c")
                            elif line.startswith("d)"):
                                option_labels["d"] = line[2:].strip()
                                options.append("d")
                
                # Make sure we have all options a, b, c, d
                expected_options = ["a", "b", "c", "d"]
                
                if len(options) > 0:
                    # Create radio buttons for options
                    selected_option = st.radio(
                        f"Select answer for question {i+1}:",
                        ["", "a", "b", "c", "d"],
                        format_func=lambda x: f"{x}) {option_labels.get(x, '')}" if x else "Select an answer...",
                        key=f"q_{i}_answer",
                        index=0
                    )
                    
                    # Store the selected answer
                    if selected_option:
                        st.session_state.quiz_user_answers[i] = selected_option
                else:
                    st.error(f"Question {i+1} has an invalid format. Please regenerate the quiz.")
                
                st.markdown("---")
            
            # Submit button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Submit Quiz"):
                    # Check if all questions have been answered
                    unanswered = [i+1 for i, ans in enumerate(st.session_state.quiz_user_answers) if not ans]
                    
                    if unanswered:
                        st.warning(f"Please answer all questions. Missing answers for questions: {', '.join(map(str, unanswered))}")
                    else:
                        # Calculate score
                        score = 0
                        for i, (user_ans, correct_ans) in enumerate(zip(st.session_state.quiz_user_answers, st.session_state.quiz_answer_key)):
                            if user_ans.lower() == correct_ans.lower():
                                score += 1
                        
                        # Calculate percentage
                        total_questions = len(st.session_state.quiz_questions)
                        percentage = (score / total_questions) * 100
                        
                        # Store results in session state
                        st.session_state.quiz_score = score
                        st.session_state.quiz_percentage = percentage
                        
                        # Show results directly without analysis for now
                        st.success(f"Quiz submitted! Your score: {score}/{total_questions} ({int(percentage)}%)")
                        
                        # Save results to database
                        try:
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            cursor.execute("""
                                INSERT INTO quizzes (
                                    student_id, title, content_source, subject, difficulty,
                                    num_questions, questions, answers, user_answers,
                                    score, score_percentage, metadata
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (
                                st.session_state['user_id'],
                                st.session_state.quiz_title,
                                st.session_state.quiz_content_source,
                                st.session_state.quiz_subject,
                                st.session_state.quiz_difficulty,
                                len(st.session_state.quiz_questions),
                                json.dumps(st.session_state.quiz_questions),
                                json.dumps(st.session_state.quiz_answer_key),
                                json.dumps(st.session_state.quiz_user_answers),
                                score,
                                percentage,
                                json.dumps({
                                    "created_at": datetime.now().isoformat()
                                })
                            ))
                            
                            quiz_id = cursor.fetchone()[0]
                            st.session_state.last_quiz_id = quiz_id
                            
                            conn.commit()
                            cursor.close()
                            conn.close()
                            
                            st.success(f"Quiz results saved to your history!")
                            
                        except Exception as db_error:
                            st.error(f"Error saving quiz results: {str(db_error)}")
            
            with col2:
                if st.button("Clear and Start Over"):
                    # Clear session state variables related to the current quiz
                    for key in ['quiz_questions', 'quiz_answer_key', 'quiz_user_answers',
                                'quiz_content', 'quiz_title', 'quiz_subject', 
                                'quiz_difficulty', 'quiz_content_source']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.experimental_rerun()
    
    with tab2:
        st.subheader("Your Quiz History")
        
        # Get quiz history from database
        conn = init_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, title, subject, difficulty, num_questions, 
                       score, score_percentage, created_at
                FROM quizzes
                WHERE student_id = %s
                ORDER BY created_at DESC
            """, (st.session_state['user_id'],))
            
            quizzes = cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving quiz history: {str(e)}")
            quizzes = []
        finally:
            cursor.close()
            conn.close()
        
        if quizzes:
            # Show summary table
            quiz_data = []
            for quiz in quizzes:
                quiz_id, title, subject, difficulty, num_q, score, percentage, date = quiz
                quiz_data.append({
                    "ID": quiz_id,
                    "Title": title,
                    "Subject": subject,
                    "Difficulty": difficulty,
                    "Questions": num_q,
                    "Score": f"{score}/{num_q}",
                    "Percentage": f"{percentage:.1f}%",
                    "Date": date.strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(quiz_data)
            st.dataframe(df, use_container_width=True)
            
            # Select a quiz to view details
            selected_quiz = st.selectbox(
                "Select a quiz to view details:",
                options=[f"{q[1]} ({q[7].strftime('%Y-%m-%d')})" for q in quizzes],
                index=0
            )
            
            selected_index = [f"{q[1]} ({q[7].strftime('%Y-%m-%d')})" for q in quizzes].index(selected_quiz)
            selected_quiz_id = quizzes[selected_index][0]
            
            # Get full quiz details
            conn = init_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT title, subject, difficulty, num_questions, 
                           questions, answers, user_answers,
                           score, score_percentage, created_at
                    FROM quizzes
                    WHERE id = %s AND student_id = %s
                """, (selected_quiz_id, st.session_state['user_id']))
                
                quiz_details = cursor.fetchone()
            except Exception as e:
                st.error(f"Error retrieving quiz details: {str(e)}")
                quiz_details = None
            finally:
                cursor.close()
                conn.close()
            
            if quiz_details:
                title, subject, difficulty, num_q, questions_json, answers_json, user_answers_json, score, percentage, date = quiz_details
                
                # Parse JSON data
                try:
                    questions = json.loads(questions_json) if isinstance(questions_json, (str, bytes, bytearray)) else questions_json
                    answers = json.loads(answers_json) if isinstance(answers_json, (str, bytes, bytearray)) else answers_json
                    user_answers = json.loads(user_answers_json) if isinstance(user_answers_json, (str, bytes, bytearray)) else user_answers_json
                except Exception as e:
                    st.error(f"Error parsing quiz data: {str(e)}")
                    questions, answers, user_answers = [], [], []
                
                # Display quiz details
                st.markdown(f"### {title}")
                st.markdown(f"**Subject:** {subject} | **Difficulty:** {difficulty} | **Date:** {date.strftime('%Y-%m-%d %H:%M')}")
                
                # Display score
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### Score: {score}/{num_q}")
                    
                    # Score gauge using progress bar
                    color = "green" if percentage >= 80 else "orange" if percentage >= 60 else "red"
                    st.markdown(
                        f"""
                        <div style="text-align: center; font-size: 24px; font-weight: bold; color: {color};">
                            {percentage:.1f}%
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.progress(float(percentage)/100)
                
                # Display questions and answers
                with st.expander("View Questions and Answers"):
                    for i, (question, user_ans, correct_ans) in enumerate(zip(questions, user_answers, answers)):
                        q_text = question.split("a)")[0].strip()
                        
                        st.markdown(f"**{i+1}. {q_text}**")
                        
                        # Extract options using improved parsing
                        options = []
                        option_labels = {}
                        
                        # More robust option extraction that handles various formatting
                        option_pattern = re.compile(r'([a-d])\)(.*?)(?=\s*[a-d]\)|$)', re.DOTALL)
                        
                        # Find all options in the question text
                        full_question_text = question
                        matches = option_pattern.findall(full_question_text)
                        
                        # Format and display options with correct/incorrect indicators
                        for opt_letter, opt_text in matches:
                            opt_letter = opt_letter.lower()
                            opt_text = opt_text.strip()
                            option_text = f"{opt_letter}) {opt_text}"
                            
                            if opt_letter == correct_ans:
                                st.markdown(f"✅ {option_text}")
                            elif opt_letter == user_ans and user_ans != correct_ans:
                                st.markdown(f"❌ {option_text}")
                            else:
                                st.markdown(f"   {option_text}")
                        
                        # Fallback to old method if no options were found
                        if not matches:
                            if "a)" in question:
                                options_text = question.split("a)")[1]
                                options_lines = options_text.split("\n")
                                
                                for line in options_lines:
                                    line = line.strip()
                                    if line.startswith(("a)", "b)", "c)", "d)")):
                                        option_letter = line[0]
                                        if option_letter == correct_ans:
                                            st.markdown(f"✅ {line}")
                                        elif option_letter == user_ans and user_ans != correct_ans:
                                            st.markdown(f"❌ {line}")
                                        else:
                                            st.markdown(f"   {line}")
                        
                        st.markdown("---")
        else:
            st.info("You haven't taken any quizzes yet. Create a quiz to see your history here.")

def dashboard_page():
    st.title("📊 Academic Analytics Dashboard")
    
    st.markdown("""
    This dashboard provides a comprehensive overview of all students in the system,
    their activities, performance metrics, and trends over time.
    """)
    
    # Check if user has admin privileges (for now, let's just allow access for all users)
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Create tabs for different views
    overview_tab, students_tab, activity_tab, performance_tab = st.tabs([
        "System Overview", "Student Leaderboard", "Activity Metrics", "Performance Analytics"
    ])
    
    # Get database connection
    conn = init_connection()
    cursor = conn.cursor()
    
    # Get total number of students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]
    
    # Get total number of notes
    cursor.execute("SELECT COUNT(*) FROM notes")
    total_notes = cursor.fetchone()[0]
    
    # Get total number of tasks
    cursor.execute("SELECT COUNT(*) FROM tasks")
    total_tasks = cursor.fetchone()[0]
    
    # Get total number of quizzes
    cursor.execute("""
        SELECT COUNT(*) FROM quizzes
    """)
    total_quizzes = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
    
    # System Overview Tab
    with overview_tab:
        st.header("System Overview")
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", total_students)
        with col2:
            st.metric("Total Notes", total_notes)
        with col3:
            st.metric("Total Tasks", total_tasks)
        with col4:
            st.metric("Total Quizzes", total_quizzes)
        
        # Activity over time
        st.subheader("System Activity Over Time")
        
        # Get activity by date (last 30 days)
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM notes
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        """)
        
        notes_activity = cursor.fetchall()
        
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM tasks
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        """)
        
        tasks_activity = cursor.fetchall()
        
        # Create activity dataframe
        if notes_activity or tasks_activity:
            activity_dates = set()
            notes_dict = {}
            tasks_dict = {}
            
            for date, count in notes_activity:
                activity_dates.add(date)
                notes_dict[date] = count
            
            for date, count in tasks_activity:
                activity_dates.add(date)
                tasks_dict[date] = count
            
            activity_data = []
            for date in sorted(activity_dates):
                activity_data.append({
                    "Date": date,
                    "Notes Created": notes_dict.get(date, 0),
                    "Tasks Created": tasks_dict.get(date, 0)
                })
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                
                # Plot activity chart
                st.line_chart(
                    activity_df.set_index("Date")[["Notes Created", "Tasks Created"]]
                )
        else:
            st.info("No activity data available for the past 30 days.")
        
        # Course subject distribution
        st.subheader("Subject Distribution")
        
        cursor.execute("""
            SELECT 
                subject,
                COUNT(*) as count
            FROM notes
            WHERE subject IS NOT NULL AND subject != ''
            GROUP BY subject
            ORDER BY count DESC
            LIMIT 10
        """)
        
        subject_data = cursor.fetchall()
        
        if subject_data:
            subject_df = pd.DataFrame(subject_data, columns=["Subject", "Count"])
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(subject_df["Subject"], subject_df["Count"])
            
            # Add labels and title
            ax.set_xlabel("Subject")
            ax.set_ylabel("Number of Notes")
            ax.set_title("Most Popular Subjects")
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.info("No subject data available.")
    
    # Student Leaderboard Tab
    with students_tab:
        st.header("Student Leaderboard")
        
        # Get student data
        cursor.execute("""
            SELECT 
                s.id,
                s.name,
                s.learning_style,
                COUNT(DISTINCT n.id) as notes_count,
                COUNT(DISTINCT t.id) as tasks_count,
                COUNT(DISTINCT CASE WHEN t.status = 'completed' THEN t.id END) as completed_tasks
            FROM 
                students s
                LEFT JOIN notes n ON s.id = n.student_id
                LEFT JOIN tasks t ON s.id = t.student_id
            GROUP BY 
                s.id, s.name, s.learning_style
            ORDER BY 
                notes_count DESC, completed_tasks DESC
        """)
        
        student_data = cursor.fetchall()
        
        # Get quiz performance
        quiz_data = {}
        try:
            cursor.execute("""
                SELECT 
                    student_id,
                    AVG(score_percentage) as avg_score,
                    COUNT(*) as quiz_count
                FROM quizzes
                GROUP BY student_id
            """)
            
            for student_id, avg_score, quiz_count in cursor.fetchall():
                quiz_data[student_id] = {
                    "avg_score": avg_score,
                    "quiz_count": quiz_count
                }
        except:
            pass  # Handle case where quizzes table doesn't exist
        
        # Create student leaderboard dataframe
        if student_data:
            leaderboard_data = []
            for student_id, name, learning_style, notes_count, tasks_count, completed_tasks in student_data:
                # Calculate productivity score
                productivity = notes_count * 5 + completed_tasks * 3
                
                # Get quiz data if available
                avg_quiz_score = 0
                quiz_count = 0
                if student_id in quiz_data:
                    avg_quiz_score = float(quiz_data[student_id]["avg_score"])
                    quiz_count = quiz_data[student_id]["quiz_count"]
                
                # Calculate overall score
                overall_score = productivity
                if quiz_count > 0:
                    overall_score += avg_quiz_score * 0.2
                
                leaderboard_data.append({
                    "ID": student_id,
                    "Name": name,
                    "Learning Style": learning_style or "Not specified",
                    "Notes": notes_count,
                    "Tasks": tasks_count,
                    "Completed Tasks": completed_tasks,
                    "Quizzes Taken": quiz_count,
                    "Avg. Quiz Score": f"{avg_quiz_score:.1f}%" if quiz_count > 0 else "N/A",
                    "Productivity Score": productivity,
                    "Overall Score": int(overall_score)
                })
            
            # Sort by overall score
            leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="Overall Score", ascending=False)
            
            # Reset index to start from 1 and rename the index column
            leaderboard_df = leaderboard_df.reset_index(drop=True)
            leaderboard_df.index = leaderboard_df.index + 1  # Make index start from 1 instead of 0
            
            # Display leaderboard with proper index column name
            st.dataframe(leaderboard_df, use_container_width=True, hide_index=False)
            
            # Highlight top performers
            if len(leaderboard_df) > 0:
                st.subheader("🏆 Top Performers")
                
                top3_cols = st.columns(min(3, len(leaderboard_df)))
                
                for i, (_, row) in enumerate(leaderboard_df.head(3).iterrows()):
                    with top3_cols[i]:
                        st.markdown(f"### {i+1}. {row['Name']}")
                        st.info(f"Score: **{row['Overall Score']}**")
                        st.caption(f"Notes: {row['Notes']} | Tasks: {row['Tasks']} | Completed: {row['Completed Tasks']}")
        else:
            st.info("No student data available.")
    
    # Activity Metrics Tab
    with activity_tab:
        st.header("Student Activity Metrics")
        
        # Select a specific student to analyze
        cursor.execute("SELECT id, name FROM students ORDER BY name")
        students = cursor.fetchall()
        
        if students:
            student_options = ["All Students"] + [f"{name} (ID: {id})" for id, name in students]
            selected_student = st.selectbox("Select Student:", student_options)
            
            # Filter condition based on selection
            if selected_student == "All Students":
                student_filter = ""
                student_id_param = None
            else:
                student_id = selected_student.split("(ID: ")[1].split(")")[0]
                student_filter = "WHERE student_id = %s"
                student_id_param = int(student_id)
            
            # Notes activity over time
            st.subheader("Notes Creation Activity")
            
            if student_id_param:
                cursor.execute(f"""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as count
                    FROM notes
                    {student_filter}
                    GROUP BY DATE(created_at)
                    ORDER BY DATE(created_at)
                """, (student_id_param,) if student_id_param else ())
            else:
                cursor.execute(f"""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as count
                    FROM notes
                    GROUP BY DATE(created_at)
                    ORDER BY DATE(created_at)
                """)
                
            notes_by_date = cursor.fetchall()
            
            if notes_by_date:
                notes_df = pd.DataFrame(notes_by_date, columns=["Date", "Notes Created"])
                st.line_chart(notes_df.set_index("Date"))
            else:
                st.info("No notes activity data available.")
            
            # Task completion metrics
            st.subheader("Task Completion Metrics")
            
            if student_id_param:
                cursor.execute(f"""
                    SELECT 
                        status,
                        COUNT(*) as count
                    FROM tasks
                    {student_filter}
                    GROUP BY status
                """, (student_id_param,) if student_id_param else ())
            else:
                cursor.execute(f"""
                    SELECT 
                        status,
                        COUNT(*) as count
                    FROM tasks
                    GROUP BY status
                """)
                
            task_status = cursor.fetchall()
            
            if task_status:
                status_df = pd.DataFrame(task_status, columns=["Status", "Count"])
                
                # Create pie chart
                fig, ax = plt.subplots()
                ax.pie(status_df["Count"], labels=status_df["Status"], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
                st.pyplot(fig)
            else:
                st.info("No task status data available.")
        else:
            st.info("No students available.")
    
    # Performance Analytics Tab
    with performance_tab:
        st.header("Learning Performance Analytics")
        
        # Quiz performance over time
        st.subheader("Quiz Performance Trends")
        
        try:
            # Select a specific student to analyze
            cursor.execute("SELECT id, name FROM students ORDER BY name")
            students = cursor.fetchall()
            
            if students:
                student_options = ["All Students"] + [f"{name} (ID: {id})" for id, name in students]
                selected_quiz_student = st.selectbox("Select Student for Quiz Analysis:", student_options, key="quiz_student")
                
                # Filter condition based on selection
                if selected_quiz_student == "All Students":
                    quiz_student_filter = ""
                    quiz_student_id_param = None
                else:
                    quiz_student_id = selected_quiz_student.split("(ID: ")[1].split(")")[0]
                    quiz_student_filter = "WHERE student_id = %s"
                    quiz_student_id_param = int(quiz_student_id)
                
                if quiz_student_id_param:
                    cursor.execute(f"""
                        SELECT 
                            created_at as date,
                            title,
                            score_percentage
                        FROM quizzes
                        {quiz_student_filter}
                        ORDER BY created_at
                    """, (quiz_student_id_param,) if quiz_student_id_param else ())
                else:
                    cursor.execute(f"""
                        SELECT 
                            created_at as date,
                            title,
                            score_percentage
                        FROM quizzes
                        ORDER BY created_at
                    """)
                    
                quiz_performance = cursor.fetchall()
                
                if quiz_performance:
                    quiz_df = pd.DataFrame(quiz_performance, columns=["Date", "Title", "Score"])
                    
                    # Convert Score from Decimal to float for calculations
                    quiz_df["Score"] = quiz_df["Score"].astype(float)
                    
                    # Create scatter plot with trend line
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(range(len(quiz_df)), quiz_df["Score"], label="Quiz Scores")
                    
                    # Add trend line if there are enough points
                    if len(quiz_df) >= 2:
                        z = np.polyfit(range(len(quiz_df)), quiz_df["Score"], 1)
                        p = np.poly1d(z)
                        ax.plot(range(len(quiz_df)), p(range(len(quiz_df))), "r--", label="Trend")
                        
                        # Determine if improving or declining
                        trend_direction = "improving" if z[0] > 0 else "declining"
                        st.info(f"Overall quiz performance is **{trend_direction}**. Trend slope: {z[0]:.2f}% per quiz.")
                    
                    # Set labels and ticks
                    ax.set_xlabel("Quiz Number")
                    ax.set_ylabel("Score (%)")
                    ax.set_title("Quiz Performance Over Time")
                    ax.set_xticks(range(len(quiz_df)))
                    ax.set_xticklabels([f"Quiz {i+1}" for i in range(len(quiz_df))], rotation=45)
                    ax.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display table of quiz results
                    st.subheader("Quiz Results")
                    formatted_quiz_df = quiz_df.copy()
                    formatted_quiz_df["Date"] = formatted_quiz_df["Date"].dt.strftime("%Y-%m-%d %H:%M")
                    formatted_quiz_df["Score"] = formatted_quiz_df["Score"].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(formatted_quiz_df, use_container_width=True)
                else:
                    st.info("No quiz performance data available.")
            else:
                st.info("No students available.")
        except Exception as e:
            st.error(f"Error analyzing quiz performance: {str(e)}")
        
        # Subject proficiency (based on quiz scores by subject)
        st.subheader("Subject Proficiency")
        
        try:
            # Get average scores by subject
            cursor.execute("""
                SELECT 
                    subject,
                    AVG(score_percentage) as avg_score,
                    COUNT(*) as quiz_count
                FROM quizzes
                GROUP BY subject
                HAVING COUNT(*) > 0
                ORDER BY avg_score DESC
            """)
            
            subject_scores = cursor.fetchall()
            
            if subject_scores:
                subject_prof_df = pd.DataFrame(subject_scores, columns=["Subject", "Average Score", "Quiz Count"])
                
                # Format columns
                subject_prof_df["Average Score"] = subject_prof_df["Average Score"].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(subject_prof_df, use_container_width=True)
                
                # Create bar chart for subject proficiency
                chart_data = pd.DataFrame(subject_scores, columns=["Subject", "Average Score", "Quiz Count"])
                
                st.bar_chart(chart_data.set_index("Subject")["Average Score"])
            else:
                st.info("No subject proficiency data available.")
        except Exception as e:
            st.error(f"Error analyzing subject proficiency: {str(e)}")
    
    # Add a new section for user management below all tabs
    st.markdown("---")
    st.header("🔒 User Management")
    
    # Create expandable section for dangerous operations
    with st.expander("Delete Student Profile"):
        st.warning("⚠️ **WARNING**: Deleting a student will permanently remove all their data including notes, tasks, quizzes, and other related information. This action cannot be undone.")
        
        # Get list of students
        cursor.execute("SELECT id, name FROM students ORDER BY name")
        students_list = cursor.fetchall()
        
        if students_list:
            # Create dropdown to select student
            delete_student_options = [f"{name} (ID: {id})" for id, name in students_list]
            selected_student_to_delete = st.selectbox(
                "Select student to delete:", 
                delete_student_options,
                key="delete_student_select"
            )
            
            # Extract the student ID from the selection
            student_id_to_delete = int(selected_student_to_delete.split("(ID: ")[1].split(")")[0])
            student_name_to_delete = selected_student_to_delete.split(" (ID:")[0]
            
            # Confirmation checkbox
            confirm_delete = st.checkbox(f"I confirm that I want to delete {student_name_to_delete} and ALL their data")
            
            # Delete button
            if st.button("Delete Student", type="primary", disabled=not confirm_delete):
                try:
                    # First, display what will be deleted
                    cursor.execute("SELECT COUNT(*) FROM notes WHERE student_id = %s", (student_id_to_delete,))
                    notes_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM tasks WHERE student_id = %s", (student_id_to_delete,))
                    tasks_count = cursor.fetchone()[0]
                    
                    try:
                        cursor.execute("SELECT COUNT(*) FROM quizzes WHERE student_id = %s", (student_id_to_delete,))
                        quizzes_count = cursor.fetchone()[0]
                    except:
                        quizzes_count = 0
                    
                    st.info(f"Deleting: {notes_count} notes, {tasks_count} tasks, {quizzes_count} quizzes, and student profile.")
                    
                    # Close current connection/cursor as they might have active session parameters
                    cursor.close()
                    conn.close()
                    
                    # Create fresh connection for deletion operations
                    delete_conn = init_connection()
                    delete_cursor = delete_conn.cursor()
                    
                    # Delete all related data - using individual statements instead of transaction
                    # 1. Delete quizzes (if table exists)
                    try:
                        delete_cursor.execute("DELETE FROM quizzes WHERE student_id = %s", (student_id_to_delete,))
                        delete_conn.commit()
                    except Exception as quiz_error:
                        st.warning(f"Note: Could not delete quizzes: {str(quiz_error)}")
                    
                    # 2. Delete tasks
                    delete_cursor.execute("DELETE FROM tasks WHERE student_id = %s", (student_id_to_delete,))
                    delete_conn.commit()
                    
                    # 3. Delete notes
                    delete_cursor.execute("DELETE FROM notes WHERE student_id = %s", (student_id_to_delete,))
                    delete_conn.commit()
                    
                    # 4. Delete knowledge_base entries related to the student
                    try:
                        # Use string comparison instead of JSON operators if possible
                        delete_cursor.execute("DELETE FROM knowledge_base WHERE metadata::text LIKE %s", 
                                       (f'%"student_id": "{student_id_to_delete}"%',))
                        delete_conn.commit()
                    except Exception as kb_error:
                        st.warning(f"Note: Could not delete some knowledge base entries: {str(kb_error)}")
                    
                    # 5. Finally delete the student
                    delete_cursor.execute("DELETE FROM students WHERE id = %s", (student_id_to_delete,))
                    delete_conn.commit()
                    
                    # Close deletion connection
                    delete_cursor.close()
                    delete_conn.close()
                    
                    # Create new connection for the rest of the page
                    conn = init_connection()
                    cursor = conn.cursor()
                    
                    st.success(f"Successfully deleted student {student_name_to_delete} and all related data.")
                    
                    # Add rerun button to refresh the page
                    if st.button("Refresh Dashboard"):
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error deleting student: {str(e)}")
                    
                    # Try to ensure we have a valid connection for the rest of the page
                    try:
                        if 'delete_cursor' in locals() and delete_cursor:
                            delete_cursor.close()
                        if 'delete_conn' in locals() and delete_conn:
                            delete_conn.close()
                        
                        conn = init_connection()
                        cursor = conn.cursor()
                    except:
                        pass
        else:
            st.info("No students available.")
    
    # Close database connections
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Check if navigation is set in session state
    if 'navigation' in st.session_state:
        # Set sidebar selection based on navigation
        pass
    main() 