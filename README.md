# Academic AI Assistant

Academic AI Assistant is a powerful multi-agent system designed to transform the way students manage their academic life. Using LangChain's advanced RAG capabilities, it creates a network of specialized AI agents that work together to provide personalized academic support.

## Features

- **Home**: Central dashboard, student profile management
- **Notewriter**: Processing academic content into personalized study materials using AI
- **Planner**: Schedule optimization and task management
- **Advisor**: Personalized learning advice based on your profile and academic data
- **PDF Chat**: Advanced RAG-based chat with PDFs, notes, and multi-source knowledge bases
- **Quiz & Analyze**: Generate quizzes from your study materials and get AI-powered performance feedback
- **Profile Management**: Create and manage student profiles with learning styles, academic goals, and performance metrics
- **AI-Enhanced Learning Assistant**: Chat with an AI assistant trained on educational resources
- **Notewriter Agent**: Transform content from various sources into structured study notes
- **Study Planner**: Create and manage study plans with time blocks and task prioritization
- **Performance Tracking**: Record and analyze academic performance over time
- **Resource Management**: Save and organize learning resources and references
- **Chatbot Interface**: Ask questions about your academic journey or get assistance
- **RAG-based PDF Chat**: Interact with your study materials using advanced retrieval-augmented generation
- **Multi-Source Knowledge Base**: Combine multiple documents for unified querying
- **Quiz & Test Prep**: Generate practice quizzes from your notes and study materials

## Project Structure

```
Academic-AI-Assistant/
├── academic_ai_assistant.py       # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
├── init_db.py                     # Database initialization script
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker configuration
├── run.py                         # Alternative entry point
├── src/                           # Core application code
│   ├── __init__.py
│   ├── data_manager.py            # Data management utilities
│   ├── extractors.py              # Content extraction utilities
│   ├── LLM.py                     # LLM integration and configuration
│   └── agents/                    # Specialized AI agents
│       ├── __init__.py
│       ├── planner.py             # Planner agent
│       ├── notewriter.py          # Notewriter agent
│       ├── advisor.py             # Advisor agent
│       └── coordinator.py         # Coordinator agent
```

## Requirements

- Python 3.8+
- PostgreSQL 12+ (or Docker for containerized setup)
- Groq API key (for all LLM capabilities)
- HuggingFace's all-MiniLM-L6-v2 model (automatically downloaded for embeddings)

## Installation

### Option 1: Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Academic-AI-Assistant.git
cd Academic-AI-Assistant
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL:
- Install PostgreSQL if not already installed
- Create a database named `academic_assistant`
- Update the `.env` file with your database credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=academic_assistant
DB_USER=postgres
DB_PASSWORD=postgres
GROQ_API_KEY=your_groq_api_key
```
4. Run the application:
```bash
python3 run.py
```

### Option 2: Docker Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Academic-AI-Assistant.git
cd Academic-AI-Assistant
```

2. Create `.env` file with your configurations (same as above)

3. Build and start containers:
```bash
docker-compose up --build
```

4. Access the application at `http://localhost:8501`

## Usage

After starting the application, you can:

1. Set up your student profile on the Home page
2. Use the Notewriter to create study materials from various sources
3. Manage your schedule with the Planner
4. Get personalized advice from the Advisor
5. Chat with your study materials using the PDF Chat feature
6. Test your knowledge with the Quiz & Analyze tool

## Troubleshooting

If you encounter issues:

1. Database connection problems:
- Verify PostgreSQL is running
- Check your `.env` file credentials
- Run `python init_db.py` to reset the database

2. Missing dependencies:
- Run `pip install -r requirements.txt` again
- Check for any error messages during installation

3. API key issues:
- Ensure your Groq API key is valid
- Check the `.env` file for correct formatting

## License

This project is licensed under the MIT License - see the LICENSE file for details.