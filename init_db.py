"""
Database initialization script for Academic AI Assistant

This script:
1. Creates the PostgreSQL database if it doesn't exist
2. Creates all required tables with their full schema
3. Ensures all columns exist (including any that might be added in future versions)
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Connection Settings 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    try:
        # Connect to default postgres database
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
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def create_tables():
    """Create all tables with their full schema including all columns"""
    try:
        # Connect to our database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
        # Create students table
        print("Creating/verifying 'students' table...")
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
        
        # Create tasks table
        print("Creating/verifying 'tasks' table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(id),
            title VARCHAR(255) NOT NULL,
            description TEXT,
            due_date TIMESTAMP,
            priority VARCHAR(50),
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create notes table with all columns
        print("Creating/verifying 'notes' table with all columns...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(id),
            title VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            subject VARCHAR(100),
            tags TEXT[],
            source_type VARCHAR(50),
            source_url TEXT,
            mindmap_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create knowledge_base table
        print("Creating/verifying 'knowledge_base' table...")
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
        
        # Create quizzes table
        print("Creating/verifying 'quizzes' table...")
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
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("All tables created/verified successfully.")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def main():
    """Main function to initialize the database"""
    print("Initializing Academic AI Assistant database...")
    
    # Create database
    if not create_database():
        print("Failed to create database. Exiting.")
        sys.exit(1)
    
    # Create tables with all columns
    if not create_tables():
        print("Failed to create tables. Exiting.")
        sys.exit(1)
    
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    main()