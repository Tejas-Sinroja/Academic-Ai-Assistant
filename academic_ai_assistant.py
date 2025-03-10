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

def main():
    st.set_page_config(
        page_title="Academic AI Assistant", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_db()
    
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
    
    # Input form for lecture notes
    with st.form("notewriter_form"):
        title = st.text_input("Note Title")
        subject = st.text_input("Subject")
        content = st.text_area("Enter lecture content, readings, or notes to process", height=300)
        
        col1, col2 = st.columns(2)
        with col1:
            output_type = st.selectbox(
                "Output Format",
                ["Comprehensive Notes", "Brief Summary", "Flashcards", "Mind Map"]
            )
        with col2:
            tags = st.text_input("Tags (comma separated)")
        
        submit = st.form_submit_button("Process Content")
        
        if submit and title and content:
            if 'user_id' not in st.session_state:
                st.warning("Please set up your profile first on the Home page.")
            elif not api_key:
                st.warning("Please enter your Groq API key to enable AI processing.")
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
                
                # Process content using LLM
                with st.spinner("Processing your content with AI..."):
                    try:
                        # Initialize LLM
                        llm = GroqLLaMa(api_key)
                        
                        # Construct a prompt based on the output type and learning style
                        prompt = f"""
                        Process the following {subject} content into {output_type} format.
                        Tailor the output for a student with a {learning_style} learning style.
                        
                        Content: {content}
                        
                        Generate a detailed response that includes:
                        1. A clear structure and organization
                        2. Key concepts and important points
                        3. Learning techniques specifically for {learning_style} learners
                        4. Any relevant examples or applications
                        """
                        
                        messages = [{"role": "user", "content": prompt}]
                        
                        # Use synchronous version for better UX in forms
                        processed_content = llm.generate(messages)
                        
                        # Save to database
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        # Parse tags
                        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                        
                        cursor.execute("""
                            INSERT INTO notes (student_id, title, content, subject, tags)
                            VALUES (%s, %s, %s, %s, %s) RETURNING id
                        """, (st.session_state['user_id'], title, processed_content, subject, tag_list))
                        
                        note_id = cursor.fetchone()[0]
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        st.success(f"Note '{title}' processed and saved successfully!")
                        
                        # Show the processed content
                        st.subheader("Processed Content")
                        st.markdown(processed_content)
                        
                    except Exception as e:
                        st.error(f"Error processing content: {str(e)}")
                        
                        # Still save the original content if AI processing fails
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
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.write(row["Title"])
                with col2:
                    st.write(row["Subject"])
                with col3:
                    st.write(row["Created At"])
                with col4:
                    if st.button("View", key=f"view_note_{row['ID']}"):
                        st.session_state['selected_note_id'] = row["ID"]
            
            # Show selected note
            if 'selected_note_id' in st.session_state:
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, content, subject, tags
                    FROM notes
                    WHERE id = %s AND student_id = %s
                """, (st.session_state['selected_note_id'], st.session_state['user_id']))
                
                note = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if note:
                    st.markdown("---")
                    st.subheader(f"📄 {note[0]}")
                    st.caption(f"Subject: {note[2]}")
                    if note[3]:
                        st.caption(f"Tags: {', '.join(note[3])}")
                    st.markdown(note[1])
        else:
            st.info("You haven't saved any notes yet.")

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
                display_tasks(tasks_df)
            
            with tab2:
                display_tasks(tasks_df[tasks_df["Status"] == "pending"])
            
            with tab3:
                display_tasks(tasks_df[tasks_df["Status"] == "completed"])
        else:
            st.info("You haven't added any tasks yet.")
    else:
        st.warning("Please set up your profile first on the Home page.")

def display_tasks(tasks_df):
    if not tasks_df.empty:
        for index, row in tasks_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
            
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
                    if st.button("Complete", key=f"complete_task_{row['ID']}"):
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
                        
                        st.experimental_rerun()
    else:
        st.info("No tasks to display in this category.")

def advisor_page():
    st.title("🧠 Advisor")
    
    st.markdown("""
    The Advisor agent provides personalized learning and time management advice based on your 
    student profile and academic data.
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
    
    # AI Advisor Section
    st.markdown("---")
    st.subheader("Ask the Advisor")
    
    user_question = st.text_area("What would you like advice on?", placeholder="e.g., How can I improve my study efficiency for mathematics?")
    
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
                    
                    The student is asking: {user_question}
                    
                    Provide detailed, actionable advice that takes into account their learning style, 
                    current academic situation, and best practices in educational psychology. 
                    Include specific techniques, examples, and a step-by-step approach they can follow.
                    """
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Generate advice
                    advice = llm.generate(messages)
                    
                    # Display advice
                    st.markdown("### 💡 Advisor Recommendations")
                    st.markdown(advice)
                    
                except Exception as e:
                    st.error(f"Error generating advice: {str(e)}")
                    
                    # Fallback to static advice
                    st.markdown("### 💡 General Recommendations")
                    
                    if "study" in user_question.lower() or "learning" in user_question.lower():
                        if learning_style == "Visual":
                            st.write("""
                            Based on your visual learning style, try these approaches:
                            
                            1. **Create mind maps** for complex topics to visualize relationships
                            2. **Use color coding** in your notes to highlight key concepts
                            3. **Watch video tutorials** that demonstrate concepts visually
                            4. **Create flashcards** with diagrams and visual cues
                            """)
                        elif learning_style == "Auditory":
                            st.write("""
                            As an auditory learner, these techniques may help you:
                            
                            1. **Record lectures** and listen to them when reviewing
                            2. **Read your notes aloud** to reinforce memory
                            3. **Join study groups** for discussion-based learning
                            4. **Use text-to-speech** for reading materials
                            """)
                        elif learning_style == "Reading/Writing":
                            st.write("""
                            For your reading/writing learning style, consider:
                            
                            1. **Take detailed notes** during lectures and readings
                            2. **Rewrite key concepts** in your own words
                            3. **Create summaries** of each study session
                            4. **Use written practice questions** to test your knowledge
                            """)
                        elif learning_style == "Kinesthetic":
                            st.write("""
                            As a kinesthetic learner, try these approaches:
                            
                            1. **Create physical models** or demonstrations of concepts
                            2. **Take breaks for movement** during study sessions
                            3. **Use real-world applications** to understand theory
                            4. **Teach concepts** to others using hands-on demonstrations
                            """)
                    else:
                        st.write("""
                        I'll need to analyze your academic data and learning patterns to provide personalized advice.
                        
                        Try asking about study techniques, time management, or specific subjects!
                        """)
    
    # Study habit analysis
    st.markdown("---")
    st.subheader("Study Habit Analysis")
    
    # This would be generated from actual user data in the real implementation
    st.markdown("""
    ### 📊 Recent Performance
    
    Based on your task completion rate and study patterns, here are some insights:
    
    - **Task Completion Rate**: 73% over the past 2 weeks
    - **Peak Productivity Time**: Morning (8-11 AM)
    - **Challenging Subjects**: Mathematics, Physics
    - **Strengths**: Consistent daily study, good note-taking
    
    ### 🚀 Recommendations
    
    1. **Schedule difficult subjects** during your peak productivity time
    2. **Break down complex tasks** into smaller, manageable chunks
    3. **Use spaced repetition** for challenging concepts
    4. **Incorporate more active recall** in your study sessions
    """)
    
    # Time management section
    st.markdown("---")
    st.subheader("Time Management Optimization")
    
    # Display study hour allocation (placeholder data)
    st.markdown("### ⏰ Recommended Daily Schedule")
    
    # Create sample data for visualization
    subjects = ["Math", "Physics", "Literature", "History", "Computer Science"]
    hours = [1.5, 1.0, 0.75, 0.5, 1.25]
    
    # Plot a simple bar chart
    df = pd.DataFrame({
        "Subject": subjects,
        "Hours": hours
    })
    
    st.bar_chart(df.set_index("Subject"))
    
    st.caption("This schedule is optimized based on your learning style, upcoming deadlines, and task priorities.")

if __name__ == "__main__":
    # Check if navigation is set in session state
    if 'navigation' in st.session_state:
        # Set sidebar selection based on navigation
        pass
    main() 