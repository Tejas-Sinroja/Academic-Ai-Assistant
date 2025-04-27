"""
Database schema update script for Academic AI Assistant

This script updates the existing database schema by adding missing columns to tables as needed.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Connection Settings 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

def update_notes_table():
    """Add source_type, source_url, and mindmap_content columns to the notes table if they don't exist"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
        # Check if source_type column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_type'
        """)
        
        if not cursor.fetchone():
            print("Adding 'source_type' column to notes table...")
            cursor.execute("""
                ALTER TABLE notes
                ADD COLUMN source_type VARCHAR(50)
            """)
            print("Column 'source_type' added successfully.")
        else:
            print("Column 'source_type' already exists in notes table.")
        
        # Check if source_url column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_url'
        """)
        
        if not cursor.fetchone():
            print("Adding 'source_url' column to notes table...")
            cursor.execute("""
                ALTER TABLE notes
                ADD COLUMN source_url TEXT
            """)
            print("Column 'source_url' added successfully.")
        else:
            print("Column 'source_url' already exists in notes table.")
        
        # Check if mindmap_content column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'mindmap_content'
        """)
        
        if not cursor.fetchone():
            print("Adding 'mindmap_content' column to notes table...")
            cursor.execute("""
                ALTER TABLE notes
                ADD COLUMN mindmap_content TEXT
            """)
            print("Column 'mindmap_content' added successfully.")
        else:
            print("Column 'mindmap_content' already exists in notes table.")
        
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error updating notes table: {e}")
        return False

def check_quizzes_table():
    """Check if the quizzes table exists and create it if not"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
        # Check if quizzes table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'quizzes'
            )
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Creating 'quizzes' table...")
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quizzes (
                id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(id),
                title VARCHAR(255) NOT NULL,
                content_source VARCHAR(255),
                subject VARCHAR(100),
                difficulty VARCHAR(50),
                num_questions INTEGER,
                questions JSONB,
                answers JSONB,
                user_answers JSONB,
                score INTEGER,
                score_percentage NUMERIC(5,2),
                analysis TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            print("Table 'quizzes' created successfully.")
        else:
            print("Table 'quizzes' already exists.")
        
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error checking quizzes table: {e}")
        return False

def main():
    """Main function to update the database schema"""
    print("Updating Academic AI Assistant database schema...")
    
    # Update notes table
    if not update_notes_table():
        print("Failed to update notes table. Exiting.")
        sys.exit(1)
    
    # Check quizzes table
    if not check_quizzes_table():
        print("Failed to check quizzes table. Exiting.")
        sys.exit(1)
    
    print("Database schema update completed successfully.")

if __name__ == "__main__":
    main() 