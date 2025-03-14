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
from langchain_community.document_loaders import PyPDFLoader
from src.agents.notewriter import get_notewriter
from src.agents.planner import get_planner
from src.agents.advisor import get_advisor
from src.extractors import extract_youtube_id

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
    """Check if the database schema is up to date"""
    conn = init_connection()
    cursor = conn.cursor()
    
    schema_issues = []
    
    # Check if source_type and source_url columns exist in notes table
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'notes' AND column_name IN ('source_type', 'source_url')
    """)
    
    existing_columns = [row[0] for row in cursor.fetchall()]
    
    if 'source_type' not in existing_columns or 'source_url' not in existing_columns:
        schema_issues.append("The notes table is missing the source_type or source_url columns")
    
    cursor.close()
    conn.close()
    
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
        "Advisor": "🧠"
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
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Academic AI Assistant")
    
def home_page():
    st.title("🏠 Welcome to Academic AI Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Your Personal Academic Companion
        
        Academic AI Assistant is your all-in-one solution for managing your academic life. 
        Our intelligent agents work together to provide you with personalized support:
        
        - **📝 Notewriter**: Generate study materials and summarize lectures
        - **📅 Planner**: Optimize your schedule and manage your academic calendar
        - **🧠 Advisor**: Get personalized learning and time management advice
        
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
    
    col1, col2, col3 = st.columns(3)
    
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
    
    # Add a chat interface at the bottom of the home page
    st.markdown("---")
    st.subheader("💬 Chat with Academic AI Assistant")
    
    # Get API key from .env or let user input it
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        api_key = st.text_input("Enter Groq API Key:", type="password")
    
    if api_key:
        # Initialize the LLM
        llm = GroqLLaMa(api_key)
        
        # Initialize chat history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)
        
        # User input - IMPORTANT: this must be outside containers like columns
        user_input = st.chat_input("Ask me anything about your studies...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                async def generate_response():
                    # Create messages list from chat history
                    messages = st.session_state.chat_history
                    response = await llm.agenerate(messages)
                    return response
                
                try:
                    response = asyncio.run(generate_response())
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display response
                    message_placeholder.markdown(response)
                except Exception as e:
                    message_placeholder.markdown(f"Error generating response: {str(e)}")

def notewriter_page():
    st.title("📝 Notewriter")
    
    st.markdown("""
    The Notewriter agent helps you process academic content and generate study materials 
    tailored to your learning style.
    """)
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for AI processing):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Source type selection
    st.subheader("Select Content Source")
    source_type = st.radio(
        "Choose your source type:",
        ["Text Input", "Web Page", "YouTube Video", "PDF Document"],
        horizontal=True
    )
    
    # Create a form for note creation
    with st.form("notewriter_form"):
        title = st.text_input("Note Title")
        subject = st.text_input("Subject")
        
        # Source content based on selected type
        if source_type == "Text Input":
            content = st.text_area("Enter lecture content, readings, or notes to process", height=300)
            source_url = None
            uploaded_file = None
        elif source_type == "Web Page":
            content = None
            source_url = st.text_input("Enter webpage URL:")
            st.info("The Notewriter will extract and process content from the webpage.")
            uploaded_file = None
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
        elif source_type == "PDF Document":
            content = None
            source_url = None
            uploaded_file = st.file_uploader("Upload PDF document:", type=["pdf"])
            st.info("The Notewriter will extract and process content from the PDF.")
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            output_format = st.selectbox(
                "Output Format",
                ["Comprehensive Notes", "Brief Summary", "Flashcards", "Mind Map"]
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
            elif not api_key:
                st.warning("Please enter your Groq API key to enable AI processing.")
            elif source_type == "Text Input" and not content:
                st.warning("Please enter some content to process.")
            elif source_type == "Web Page" and not source_url:
                st.warning("Please enter a webpage URL.")
            elif source_type == "YouTube Video" and not source_url:
                st.warning("Please enter a YouTube video URL.")
            elif source_type == "PDF Document" and not uploaded_file:
                st.warning("Please upload a PDF document.")
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
                
                # Import the notewriter agent
                notewriter = get_notewriter()
                
                if not notewriter:
                    st.error("Failed to initialize the Notewriter agent. Please check your Groq API key.")
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
                        elif source_type == "PDF Document":
                            # Read the uploaded file
                            source_data = uploaded_file.read()
                            source_type_code = "pdf"
                        
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
                                tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                                note_id = notewriter.add_note(
                                    st.session_state['user_id'],
                                    {
                                        "title": title,
                                        "content": content,
                                        "subject": subject,
                                        "tags": tag_list
                                    }
                                )
                                
                                if note_id:
                                    st.info("Original content saved without AI processing.")
                                    st.session_state['selected_note_id'] = note_id
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        
                        # Try to save raw content as fallback
                        if source_type == "Text Input":
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            # Parse tags
                            tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                            
                            cursor.execute("""
                                INSERT INTO notes (student_id, title, content, subject, tags)
                                VALUES (%s, %s, %s, %s, %s) RETURNING id
                            """, (st.session_state['user_id'], title, content, subject, tag_list))
                            
                            note_id = cursor.fetchone()[0]
                            conn.commit()
                            cursor.close()
                            conn.close()
                            
                            st.info("Original content saved without AI processing.")
                            st.session_state['selected_note_id'] = note_id
    
    # Display saved notes
    st.markdown("---")
    st.subheader("Your Saved Notes")
    
    if 'user_id' in st.session_state:
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
            notes_df = pd.DataFrame(notes, columns=["ID", "Title", "Subject", "Created At"])
            notes_df["Created At"] = pd.to_datetime(notes_df["Created At"]).dt.strftime("%Y-%m-%d %H:%M")
            
            # Display as table with view buttons
            for index, row in notes_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
                with col1:
                    st.write(row["Title"])
                with col2:
                    st.write(row["Subject"])
                with col3:
                    st.write(row["Created At"])
                with col4:
                    if st.button("View", key=f"view_note_{row['ID']}"):
                        st.session_state['selected_note_id'] = row["ID"]
                        # Clear delete confirmation flag if set
                        if 'delete_confirmation' in st.session_state:
                            del st.session_state['delete_confirmation']
                with col5:
                    if st.button("Delete", key=f"delete_note_{row['ID']}"):
                        st.session_state['delete_note_id'] = row["ID"]
                        st.session_state['delete_confirmation'] = False
            
            # Show delete confirmation
            if 'delete_note_id' in st.session_state and 'delete_confirmation' in st.session_state and not st.session_state['delete_confirmation']:
                note_id = st.session_state['delete_note_id']
                st.warning(f"Are you sure you want to delete this note? This action cannot be undone.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete", key=f"confirm_delete_{note_id}"):
                        # Import the notewriter agent
                        notewriter = get_notewriter()
                        
                        if notewriter and notewriter.delete_note(note_id, st.session_state['user_id']):
                            st.success("Note deleted successfully!")
                            st.session_state['delete_confirmation'] = True
                            
                            # Remove the selected note if it's the one being deleted
                            if 'selected_note_id' in st.session_state and st.session_state['selected_note_id'] == note_id:
                                del st.session_state['selected_note_id']
                            
                            # Refresh the page
                            st.rerun()
                        else:
                            st.error("Failed to delete note.")
                with col2:
                    if st.button("Cancel", key=f"cancel_delete_{note_id}"):
                        # Clear deletion state
                        del st.session_state['delete_note_id']
                        del st.session_state['delete_confirmation']
                        st.rerun()
            
            # Show selected note
            if 'selected_note_id' in st.session_state:
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, content, subject, tags, source_type, source_url, id
                    FROM notes
                    WHERE id = %s AND student_id = %s
                """, (st.session_state['selected_note_id'], st.session_state['user_id']))
                
                note = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if note:
                    st.markdown("---")
                    # Add note actions in a row
                    note_id = note[6]  # ID is at index 6 now
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.subheader(f"📄 {note[0]}")
                    with col2:
                        if st.button("Delete Note", key=f"delete_current_note_{note_id}"):
                            st.session_state['delete_note_id'] = note_id
                            st.session_state['delete_confirmation'] = False
                            st.rerun()
                    
                    st.caption(f"Subject: {note[2]}")
                    
                    # Display source information if available
                    if note[4]:  # source_type
                        source_type_display = note[4].capitalize()
                        if note[5]:  # source_url
                            st.caption(f"Source: {source_type_display} - [{note[5]}]({note[5]})")
                        else:
                            st.caption(f"Source: {source_type_display}")
                    
                    if note[3]:  # tags
                        st.caption(f"Tags: {', '.join(note[3])}")
                    
                    # Display note content
                    st.markdown(note[1])
                    
                    # Add download button
                    note_text = f"# {note[0]}\n\nSubject: {note[2]}\n"
                    if note[4]:  # source_type
                        source_type_display = note[4].capitalize()
                        if note[5]:  # source_url
                            note_text += f"Source: {source_type_display} - {note[5]}\n"
                        else:
                            note_text += f"Source: {source_type_display}\n"
                    if note[3]:  # tags
                        note_text += f"Tags: {', '.join(note[3])}\n"
                    note_text += f"\n{note[1]}"
                    
                    st.download_button(
                        label="Download Note as Markdown",
                        data=note_text,
                        file_name=f"{note[0].replace(' ', '_')}.md",
                        mime="text/markdown",
                    )
        else:
            st.info("You haven't saved any notes yet.")
    else:
        st.warning("Please set up your profile first on the Home page.")

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
    
    if st.button("Get Advice") and user_question:
        if not api_key:
            st.warning("Please enter your Groq API key to enable AI advice.")
        else:
            # Get task and note statistics for context
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
                    syllabus_metadata = json.loads(syllabus_result[1])
            
            cursor.close()
            conn.close()
            
            with st.spinner("Generating personalized advice..."):
                try:
                    # Initialize LLM
                    llm = GroqLLaMa(api_key)
                    
                    # Create a context-rich prompt
                    prompt = f"""
                    As an academic advisor, provide personalized advice for a student with the following profile:
                    
                    Learning Style: {learning_style}
                    Daily Study Hours: {study_hours}
                    
                    Task Statistics:
                    - Completed tasks: {task_stats.get('completed', 0)}
                    - Pending tasks: {task_stats.get('pending', 0)}
                    
                    Subject Distribution:
                    {', '.join([f"{subject}: {count}" for subject, count in subject_stats.items()])}
                    """
                    
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
                    
                    Provide detailed, actionable advice that takes into account their learning style, 
                    current academic situation, and best practices in educational psychology. 
                    
                    If the student has provided a course syllabus, analyze it to provide advice specific to:
                    - Important topics and concepts in the course
                    - Upcoming assignments or exams mentioned in the syllabus
                    - Recommended study techniques for the specific course material
                    - How to budget time effectively for this course
                    
                    Include specific techniques, examples, and a step-by-step approach they can follow.
                    Your advice should be concrete and practical, not general or vague.
                    """
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Generate advice
                    advice = llm.generate(messages)
                    
                    # Display advice
                    st.markdown("### 💡 Advisor Recommendations")
                    st.markdown(advice)
                    
                except Exception as e:
                    st.error(f"Error generating advice: {str(e)}")
                    st.warning("Please check your API key and try again.")
    
    # Academic Analysis Section
    st.markdown("---")
    st.subheader("Advanced Academic Analysis")
    
    # Integrate with the advisor agent to get real analysis
    if st.button("Generate Academic Analysis"):
        if not api_key:
            st.warning("Please enter your Groq API key to enable analysis.")
        else:
            with st.spinner("Analyzing your academic data..."):
                try:
                    # Import the advisor agent
                    advisor = get_advisor()
                    
                    if advisor:
                        # Get advice from the advisor agent
                        advice_data = advisor.generate_advice(st.session_state['user_id'])
                        
                        if "error" in advice_data:
                            st.error(advice_data["error"])
                        else:
                            # Display actual data from the advisor agent
                            st.subheader("Study Habit Analysis")
                            
                            # Display task statistics
                            task_stats = advice_data.get("task_statistics", {})
                            if task_stats:
                                st.markdown("### 📊 Current Academic Statistics")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Tasks", task_stats.get("total_tasks", 0))
                                    st.metric("Completion Rate", 
                                             f"{task_stats.get('completion_rate_14d', 0)}%", 
                                             help="Task completion rate over the last 14 days")
                                with col2:
                                    st.metric("Completed Tasks", task_stats.get("completed_tasks", 0))
                                    st.metric("Pending Tasks", task_stats.get("pending_tasks", 0))
                            
                            # Display recommendations
                            st.markdown("### 🚀 Personalized Recommendations")
                            for advice in advice_data.get("time_management_advice", []):
                                st.markdown(f"- {advice}")
                    else:
                        st.error("Failed to initialize the Advisor agent. Please check your API key.")
                        
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")

if __name__ == "__main__":
    # Check if navigation is set in session state
    if 'navigation' in st.session_state:
        # Set sidebar selection based on navigation
        pass
    main() 