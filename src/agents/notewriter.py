"""
Notewriter Agent - Academic content processing and study material generation

This agent is responsible for:
1. Processing lecture content and readings
2. Generating comprehensive notes
3. Creating study materials based on learning style
4. Summarizing and organizing academic content
5. Researching topics automatically and generating structured notes
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2 import errors as psycopg2_errors
import json
from datetime import datetime
import tempfile
import sys
import importlib.util
from pathlib import Path
import validators

# Apply nest_asyncio to allow nested event loops (needed for Streamlit)
nest_asyncio.apply()

# Import content extractors
# First, make sure src is in the path
module_path = Path(__file__).parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
    
# Then import the extractors module
try:
    from extractors import extract_website_content, extract_pdf_content, extract_youtube_content, research_topic
except ImportError:
    # Fallback to importing with full path
    from src.extractors import extract_website_content, extract_pdf_content, extract_youtube_content, research_topic

# Load environment variables
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Define system prompts for different source types
WEB_NOTES_PROMPT = """You are an academic assistant tasked with creating detailed, structured notes from web content.
Follow these guidelines:
1. Identify the main topic and key concepts
2. Create a hierarchical structure with headings and subheadings
3. Use bullet points for key information
4. Include important definitions, statistics, and examples
5. Cite references where applicable
6. Include a brief summary at the beginning

Your notes should be comprehensive but focused on academically relevant information."""

PDF_NOTES_PROMPT = """You are an academic assistant tasked with creating detailed, structured notes from PDF content.
Follow these guidelines:
1. Identify the document type (research paper, textbook chapter, etc.)
2. Extract the main thesis or central arguments
3. Create a hierarchical structure matching the document organization
4. Include methodology and findings for research papers
5. Note definitions, theorems, and important formulas
6. Include a brief abstract/summary at the beginning

Your notes should preserve the academic rigor of the original while making it more accessible."""

YOUTUBE_NOTES_PROMPT = """You are an academic assistant tasked with creating detailed, structured notes from YouTube video content.
Follow these guidelines:
1. Start with the video title, channel, and a brief overview
2. Create a timeline of key points with timestamps where possible
3. Organize content into logical sections even if the video isn't structured that way
4. Highlight key concepts, definitions, and examples
5. Differentiate between factual content and opinions/commentary
6. Include a brief summary at the beginning

Your notes should transform the audio-visual content into well-structured written notes for academic use."""

class Notewriter:
    """Notewriter Agent for academic content processing"""
    
    def __init__(self, llm, openrouter_llm=None):
        """Initialize the notewriter agent"""
        self.conn = self._init_connection()
        self.llm = llm
        self.openrouter_llm = openrouter_llm  # For mindmap generation
    
    def _init_connection(self):
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
        except psycopg2.OperationalError as e:
            print(f"Database connection error: {e}")
            return None
    
    def get_notes(self, student_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get notes for a specific student with optional subject filter"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        
        # Construct query based on whether subject filter is provided
        if subject:
            cursor.execute("""
                SELECT id, title, content, subject, tags, created_at
                FROM notes
                WHERE student_id = %s AND subject = %s
                ORDER BY created_at DESC
            """, (student_id, subject))
        else:
            cursor.execute("""
                SELECT id, title, content, subject, tags, created_at
                FROM notes
                WHERE student_id = %s
                ORDER BY created_at DESC
            """, (student_id,))
        
        # Fetch notes and format as dictionary
        notes = []
        for row in cursor.fetchall():
            notes.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "subject": row[3],
                "tags": row[4],
                "created_at": row[5]
            })
        
        cursor.close()
        return notes
    
    def get_note_by_id(self, note_id: int, student_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific note by ID"""
        if not self.conn:
            return None
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, title, content, subject, tags, created_at, mindmap_content
            FROM notes
            WHERE id = %s AND student_id = %s
        """, (note_id, student_id))
        
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "subject": row[3],
            "tags": row[4],
            "created_at": row[5],
            "mindmap_content": row[6]
        }
    
    def add_note(self, student_id, note_data):
        """
        Add a note to the database
        
        Args:
            student_id (int): The student ID
            note_data (dict): Note data including title, content, subject, tags, 
                              source_type, and source_url
        
        Returns:
            int: The note ID if successful, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Extract note data
            title = note_data.get('title')
            content = note_data.get('content')
            subject = note_data.get('subject')
            tags = note_data.get('tags', [])
            source_type = note_data.get('source_type')
            source_url = note_data.get('source_url')
            
            # Check if the notes table has the source columns
            try:
                # Try inserting with source columns
                cursor.execute("""
                    INSERT INTO notes 
                    (student_id, title, content, subject, tags, source_type, source_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (student_id, title, content, subject, tags, source_type, source_url))
            except psycopg2_errors.UndefinedColumn:
                # If source columns don't exist, try without them
                print("Warning: source_type or source_url columns don't exist. Using fallback query.")
                cursor.execute("""
                    INSERT INTO notes 
                    (student_id, title, content, subject, tags)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (student_id, title, content, subject, tags))
                
                # Suggest to the user to run update_db_schema.py
                print("Please run 'python update_db_schema.py' to update the database schema.")
            
            note_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            
            return note_id
        except Exception as e:
            print(f"Error adding note: {str(e)}")
            # Try a simple fallback insertion if all else fails
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO notes 
                    (student_id, title, content, subject)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (student_id, title, content, subject))
                
                note_id = cursor.fetchone()[0]
                self.conn.commit()
                cursor.close()
                
                return note_id
            except Exception as inner_e:
                print(f"Fallback insertion also failed: {str(inner_e)}")
                return None
    
    def update_note(self, note_id: int, student_id: int, note_data: Dict[str, Any]) -> bool:
        """Update an existing note"""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        
        # Build dynamic update query based on provided data
        update_fields = []
        params = []
        
        if "title" in note_data:
            update_fields.append("title = %s")
            params.append(note_data["title"])
        
        if "content" in note_data:
            update_fields.append("content = %s")
            params.append(note_data["content"])
        
        if "subject" in note_data:
            update_fields.append("subject = %s")
            params.append(note_data["subject"])
        
        if "tags" in note_data:
            update_fields.append("tags = %s")
            # Parse tags if they're provided as a string
            tags = note_data["tags"]
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            params.append(tags)
        
        # Always update the updated_at timestamp
        update_fields.append("updated_at = NOW()")
        
        # If no fields to update, return early
        if not update_fields:
            return False
        
        # Add note_id and student_id to params
        params.extend([note_id, student_id])
        
        try:
            cursor.execute(f"""
                UPDATE notes 
                SET {", ".join(update_fields)}
                WHERE id = %s AND student_id = %s
            """, params)
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            return affected_rows > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating note: {e}")
            return False
        finally:
            cursor.close()
    
    def delete_note(self, note_id: int, student_id: int) -> bool:
        """Delete a note"""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM notes
                WHERE id = %s AND student_id = %s
            """, (note_id, student_id))
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            return affected_rows > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting note: {e}")
            return False
        finally:
            cursor.close()
    
    def search_notes(self, student_id: int, query: str) -> List[Dict[str, Any]]:
        """Search notes by content for a student"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        
        # Use PostgreSQL full-text search capabilities
        cursor.execute("""
            SELECT id, title, content, subject, tags, created_at,
                   ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s)) as rank
            FROM notes
            WHERE student_id = %s 
              AND (to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %s)
                   OR %s = ANY(tags))
            ORDER BY rank DESC
        """, (query, student_id, query, query))
        
        # Fetch notes and format as dictionary
        notes = []
        for row in cursor.fetchall():
            notes.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "subject": row[3],
                "tags": row[4],
                "created_at": row[5],
                "relevance": row[6]
            })
        
        cursor.close()
        return notes
    
    async def extract_content(self, source_type: str, source: str) -> Union[str, Tuple[bool, str]]:
        """
        Extract content from various sources
        
        Args:
            source_type (str): Type of source ('web', 'pdf', 'youtube', 'text', 'topic')
            source (str): URL, file path, or raw text
            
        Returns:
            Union[str, Tuple[bool, str]]: Either extracted content or a tuple (False, error_message)
        """
        try:
            if source_type == "web":
                return await extract_website_content(source)
            elif source_type == "pdf":
                # For uploaded files, we'll need to handle bytes instead of a path
                if isinstance(source, bytes):
                    return extract_pdf_content(source)
                return extract_pdf_content(source)
            elif source_type == "youtube":
                # Validate URL before attempting to extract content
                if not validators.url(source):
                    return (False, f"Invalid YouTube URL: {source}")
                if "youtube.com" not in source and "youtu.be" not in source:
                    return (False, f"URL does not appear to be a YouTube link: {source}")
                return await extract_youtube_content(source)
            elif source_type == "text":
                # Direct text input - just return it
                return source
            elif source_type == "topic":
                # Process the topic using the research functionality
                depth = "deep" if source.get("depth") == "deep" else "ordinary"
                research_results = await research_topic(source.get("topic"), depth)
                return research_results["combined_content"]
            else:
                return (False, f"Unsupported source type: {source_type}")
        except Exception as e:
            return (False, f"Error extracting content from {source_type}: {str(e)}")
    
    async def process_source(self, student_id, source_type, source, title, subject, focus_area="", tags="", learning_style="Visual"):
        """
        Process the source content and generate a note
        
        Args:
            student_id (int): The student ID
            source_type (str): The type of source ('text', 'web', 'youtube', 'pdf', 'topic')
            source (str or dict): The source content, URL, or topic information
            title (str): The note title
            subject (str): The subject of the note
            focus_area (str, optional): Specific focus area within the subject
            tags (str, optional): Comma-separated tags
            learning_style (str, optional): The student's learning style
            
        Returns:
            Dict: Result information including note ID and status
        """
        try:
            # Extract content from source
            content = await self.extract_content(source_type, source)
            
            # Check if content extraction failed (returns a tuple)
            if isinstance(content, tuple) and len(content) == 2 and content[0] is False:
                return {
                    "success": False,
                    "error": content[1]  # Return the error message
                }
            
            if not content:
                return {
                    "success": False,
                    "error": f"Failed to extract content from {source_type} source"
                }
            
            # Parse tags
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            
            # Construct a prompt based on the source type, learning style and focus area
            if focus_area:
                focus_instruction = f"Pay special attention to aspects related to: {focus_area}."
            else:
                focus_instruction = ""
                
            # Build the system message based on source type
            if source_type == "web":
                system_message = f"""
                Process the following web content about {subject} into comprehensive study notes.
                {focus_instruction}
                Tailor the output for a student with a {learning_style} learning style.
                
                Create well-structured notes with:
                - Clear section headings and organization
                - Key concepts and main points highlighted
                - Explanations adapted for {learning_style} learners
                - Visual cues and organizational elements in the Markdown
                
                Source: {source}
                """
            elif source_type == "youtube":
                system_message = f"""
                Process the following YouTube video transcript about {subject} into comprehensive study notes.
                {focus_instruction}
                Tailor the output for a student with a {learning_style} learning style.
                
                Create well-structured notes that:
                1. Begin with an overview of the main topics covered in the transcript
                2. Organize the content into logical sections with clear headings
                3. Identify and highlight key concepts, definitions, and examples
                4. Create a coherent structure even if the transcript is unstructured
                5. Provide a summary of the most important points at the end
                6. Add learning recommendations specifically for {learning_style} learners
                
                Format the notes in clean Markdown, using appropriate heading levels, bullet points, 
                and emphasis to create a visually structured document.
                
                Source: {source}
                """
            elif source_type == "pdf":
                system_message = f"""
                Process the following PDF content about {subject} into comprehensive study notes.
                {focus_instruction}
                Tailor the output for a student with a {learning_style} learning style.
                
                Create well-structured notes with:
                - Preservation of the document's original structure
                - Key concepts and main points emphasized
                - Clear section headings from the original document
                - Learning techniques specifically for {learning_style} learners
                
                Format the notes in clean Markdown.
                """
            elif source_type == "topic":
                # For topic-based research, we have a more specialized prompt
                system_message = f"""
                Process the following research on {subject} into comprehensive study notes with proper academic citations.
                {focus_instruction}
                Tailor the output for a student with a {learning_style} learning style.

                This research includes content from multiple sources including web pages and YouTube videos.

                Create well-structured, comprehensive study notes that:
                1. Begin with an overview of {subject} and research methodology
                2. Organize content into logical sections with clear headings
                3. Clearly identify and cite sources for all key information:
                   - For web sources: [Source #](URL) - Title
                   - For videos: [Video #](URL) - Title
                4. Synthesize information while maintaining clear attribution
                5. Include direct quotes with proper citation when appropriate
                6. Highlight key concepts, definitions, and examples with sources
                7. End with a summary and bibliography listing all sources

                Citation Format Examples:
                Web Sources:
                - "According to [Web Source 1](https://example.com) titled 'Article Name', the key concept is..."
                - "As reported by [Web Source 2](https://example.org), researchers found that..."
                
                Video Sources:
                - "As demonstrated in [Video 1](https://youtube.com/watch?v=123) 'Video Title', the process involves..."
                - "The experiment shown in [Video 2](https://youtube.com/watch?v=456) illustrates..."

                Format requirements:
                - Use Markdown with clear headings and bullet points
                - Include a References section at the end listing all sources
                - Maintain academic integrity by properly attributing all sources
                - Adapt content presentation for {learning_style} learners

                Important: Failure to properly cite sources will result in academic dishonesty.
                """
            else:  # text
                system_message = f"""
                Process the following {subject} content into comprehensive study notes.
                {focus_instruction}
                Tailor the output for a student with a {learning_style} learning style.
                
                Generate detailed notes that include:
                - A clear structure and organization
                - Key concepts and important points
                - Learning techniques specifically for {learning_style} learners
                - Any relevant examples or applications
                
                Format the notes in clean Markdown.
                """
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": content}
            ]
            
            # Generate notes
            processed_content = self.llm.generate(messages)
            
            # Determine source URL for storage
            source_url = None
            if source_type in ["web", "youtube"]:
                source_url = source
            elif source_type == "topic" and isinstance(source, dict):
                source_url = f"Research on: {source.get('topic')}"
            
            # Save the note
            note_data = {
                "title": title,
                "content": processed_content,
                "subject": subject,
                "tags": tag_list,
                "source_type": source_type,
                "source_url": source_url
            }
            
            note_id = self.add_note(student_id, note_data)
            
            if note_id:
                return {
                    "success": True,
                    "note_id": note_id,
                    "content": processed_content  # Return the content for mindmap generation
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save note to database"
                }
                
        except Exception as e:
            print(f"Error in process_source: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_topic(self, student_id, topic, search_depth, title, subject, focus_area="", tags="", learning_style="Visual"):
        """
        Research a topic and generate comprehensive notes
        
        Args:
            student_id (int): The student ID
            topic (str): The topic to research
            search_depth (str): "ordinary" or "deep" search
            title (str): The note title
            subject (str): The subject of the note
            focus_area (str, optional): Specific focus area within the subject
            tags (str, optional): Comma-separated tags
            learning_style (str, optional): The student's learning style
            
        Returns:
            Dict: Result information including note ID and status
        """
        try:
            # Prepare the topic source information
            source = {
                "topic": topic,
                "depth": search_depth
            }
            
            # Process the topic as a source
            result = await self.process_source(
                student_id=student_id,
                source_type="topic",
                source=source,
                title=title,
                subject=subject,
                focus_area=focus_area,
                tags=tags,
                learning_style=learning_style
            )
            
            return result
        except Exception as e:
            print(f"Error in process_topic: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_mindmap(self, content: str) -> str:
        """
        Generate a mindmap from note content
        
        Args:
            content (str): The note content to generate a mindmap for
            
        Returns:
            str: A markdown-formatted mindmap
        """
        try:
            # Always use the primary LLM
            llm_to_use = self.llm
            
            # Create a system prompt for mindmap generation
            system_message = """
            Convert the following notes into a structured markdown mindmap.
            
            Format the mindmap as a Markdown list with hierarchy showing the relationships 
            between concepts. The output should be formatted as follows:
            
            # Main Topic
            
            ## Major Concept 1
            - Sub-concept 1.1
                - Detail 1.1.1
                - Detail 1.1.2
            - Sub-concept 1.2
            
            ## Major Concept 2
            - Sub-concept 2.1
            - Sub-concept 2.2
                - Detail 2.2.1
            
            Make sure:
            1. The hierarchy clearly shows relationships between concepts
            2. The mindmap accurately summarizes the main ideas and details
            3. The format is valid markdown using headings and nested lists
            4. You include ALL important information from the notes
            5. You DO NOT add any new information that isn't in the notes
            
            This will be visualized as a mindmap, so proper indentation and hierarchy are critical.
            """
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": content}
            ]
            
            # Generate mindmap content
            mindmap_content = llm_to_use.generate(messages)
            
            return mindmap_content
        except Exception as e:
            print(f"Error generating mindmap: {str(e)}")
            # Return a basic mindmap structure if generation fails
            return f"# {content[:50]}...\n\n## Failed to generate detailed mindmap\n- Error: {str(e)}"
    
    def save_mindmap(self, note_id: int, mindmap_content: str) -> bool:
        """
        Save a mindmap outline to the database for a specific note
        
        Args:
            note_id (int): The ID of the note to save the mindmap for
            mindmap_content (str): The markdown-formatted mindmap content
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE notes
                SET mindmap_content = %s
                WHERE id = %s
            """, (mindmap_content, note_id))
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            cursor.close()
            
            return affected_rows > 0
        except Exception as e:
            print(f"Error saving mindmap: {e}")
            self.conn.rollback()
            return False
    
    def get_mindmap(self, note_id: int) -> Optional[str]:
        """
        Retrieve a mindmap outline from the database for a specific note
        
        Args:
            note_id (int): The ID of the note to get the mindmap for
            
        Returns:
            Optional[str]: The mindmap content or None if not found
        """
        if not self.conn:
            return None
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT mindmap_content
                FROM notes
                WHERE id = %s
            """, (note_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0]:
                return result[0]
            return None
        except Exception as e:
            print(f"Error retrieving mindmap: {e}")
            return None
            
    # Close the database connection when done
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# Add helper function to get the notewriter instance
def get_notewriter(llm_type="groq", groq_api_key=None, openrouter_api_key=None, openrouter_model=None):
    """
    Returns an instance of the Notewriter agent.
    This function is used by the Streamlit app to get a notewriter instance.
    
    Args:
        llm_type (str): The type of LLM to use ("groq" or "openrouter")
        groq_api_key (str, optional): API key for Groq
        openrouter_api_key (str, optional): API key for OpenRouter
        openrouter_model (str, optional): Model name for OpenRouter
    
    Returns:
        Notewriter: An instance of the Notewriter agent
    """
    # Initialize the LLM
    from src.LLM import GroqLLaMa, OpenRouterLLM, LLMConfig
    import os
    
    # If API keys not provided, get from environment
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    # Initialize LLMs based on type requested
    groq_llm = None
    openrouter_llm = None
    
    if llm_type == "groq":
        # Check if we have a valid Groq API key
        if not groq_api_key or groq_api_key == "your_groq_api_key":
            return None
        
        # Initialize Groq LLM
        groq_llm = GroqLLaMa(groq_api_key)
        
        # If we also have OpenRouter key, initialize it as secondary LLM
        if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key":
            openrouter_llm = OpenRouterLLM(openrouter_api_key)
            if openrouter_model:
                openrouter_llm.config.openrouter_model = openrouter_model
                
        # Return Notewriter with Groq as primary LLM
        return Notewriter(groq_llm, openrouter_llm)
        
    elif llm_type == "openrouter":
        # Check if we have a valid OpenRouter API key
        if not openrouter_api_key or openrouter_api_key == "your_openrouter_api_key":
            return None
            
        # Initialize OpenRouter LLM
        openrouter_llm = OpenRouterLLM(openrouter_api_key)
        
        # Set custom model if provided
        if openrouter_model:
            openrouter_llm.config.openrouter_model = openrouter_model
            
        # If we also have Groq key, initialize it as secondary LLM
        if groq_api_key and groq_api_key != "your_groq_api_key":
            groq_llm = GroqLLaMa(groq_api_key)
            
        # Return Notewriter with OpenRouter as primary LLM and ability to generate mindmaps
        return Notewriter(openrouter_llm, openrouter_llm)
    
    # Default fallback to None if invalid LLM type
    return None 