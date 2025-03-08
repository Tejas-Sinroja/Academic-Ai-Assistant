"""
Notewriter Agent - Academic content processing and study material generation

This agent is responsible for:
1. Processing lecture content and readings
2. Generating comprehensive notes
3. Creating study materials based on learning style
4. Summarizing and organizing academic content
"""

from typing import Dict, List, Any, Optional, Tuple
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

class Notewriter:
    """Notewriter Agent for academic content processing"""
    
    def __init__(self):
        """Initialize the notewriter agent"""
        self.conn = self._init_connection()
    
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
            SELECT id, title, content, subject, tags, created_at
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
            "created_at": row[5]
        }
    
    def add_note(self, student_id: int, note_data: Dict[str, Any]) -> Optional[int]:
        """Add a new note for a student"""
        if not self.conn:
            return None
        
        cursor = self.conn.cursor()
        
        try:
            # Parse tags if they're provided as a string
            tags = note_data.get("tags", [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            
            cursor.execute("""
                INSERT INTO notes (
                    student_id, title, content, subject, tags
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                student_id,
                note_data.get("title", "Untitled Note"),
                note_data.get("content", ""),
                note_data.get("subject", ""),
                tags
            ))
            
            note_id = cursor.fetchone()[0]
            self.conn.commit()
            return note_id
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding note: {e}")
            return None
        finally:
            cursor.close()
    
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
    
    def process_content(self, content: str, student_id: int, output_format: str, 
                        learning_style: str) -> Dict[str, Any]:
        """
        Process academic content and generate study materials
        
        This is a placeholder implementation. In a real system, this would:
        1. Use LLMs to analyze and process the content
        2. Generate material based on the student's learning style
        3. Format output according to the requested format
        """
        # This is a simplified implementation - a real version would use LLMs
        
        # Generate title from content
        title = f"Notes on {content.split()[0:3]}" if content else "Untitled Notes"
        
        # Generate summary - in real implementation, this would use LLM summarization
        summary = f"Summary of the content: {content[:100]}..." if len(content) > 100 else content
        
        # Generate outline - in real implementation, this would extract key points
        outline_points = [
            "Introduction to the topic",
            "Key concept #1",
            "Key concept #2",
            "Applications and examples",
            "Conclusion and summary"
        ]
        
        # Adjust based on learning style
        style_adjustments = {}
        if learning_style == "Visual":
            style_adjustments = {
                "recommendations": [
                    "Create a mind map to visualize relationships between concepts",
                    "Use color coding for different themes in your notes",
                    "Draw diagrams to represent processes"
                ]
            }
        elif learning_style == "Auditory":
            style_adjustments = {
                "recommendations": [
                    "Record yourself reading these notes aloud",
                    "Create verbal mnemonics for key points",
                    "Discuss these concepts with study partners"
                ]
            }
        elif learning_style == "Reading/Writing":
            style_adjustments = {
                "recommendations": [
                    "Rewrite these notes in your own words",
                    "Create written summaries after each study session",
                    "Practice explaining concepts in writing"
                ]
            }
        elif learning_style == "Kinesthetic":
            style_adjustments = {
                "recommendations": [
                    "Create physical flashcards you can manipulate",
                    "Take short breaks for movement between study sessions",
                    "Role-play or act out processes when possible"
                ]
            }
        
        # Format based on requested output
        formatted_output = {}
        if output_format == "Comprehensive Notes":
            formatted_output = {
                "title": title,
                "content": content,
                "summary": summary,
                "outline": outline_points,
                "learning_recommendations": style_adjustments.get("recommendations", [])
            }
        elif output_format == "Brief Summary":
            formatted_output = {
                "title": title,
                "summary": summary,
                "key_points": outline_points,
                "learning_recommendations": style_adjustments.get("recommendations", [])
            }
        elif output_format == "Flashcards":
            # Generate example flashcards from content
            formatted_output = {
                "title": title,
                "flashcards": [
                    {"front": "What is the main topic?", "back": title},
                    {"front": "Summarize the key concept", "back": summary},
                    {"front": "List an application", "back": "This would depend on the specific content."}
                ],
                "learning_recommendations": style_adjustments.get("recommendations", [])
            }
        elif output_format == "Mind Map":
            formatted_output = {
                "title": title,
                "central_concept": title,
                "branches": outline_points,
                "learning_recommendations": style_adjustments.get("recommendations", [])
            }
        
        return formatted_output
    
    def close_connection(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

# Helper function to get a notewriter instance
def get_notewriter() -> Notewriter:
    """Get a notewriter agent instance"""
    return Notewriter() 