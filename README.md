# Academic AI Assistant

Academic AI Assistant is a powerful multi-agent system designed to transform the way students manage their academic life. Using LangGraph's workflow framework, it creates a network of specialized AI agents that work together to provide personalized academic support.

## Features

- **Home**: Central dashboard, student profile management, and AI chat interface
- **Notewriter**: Processing academic content into personalized study materials using AI
- **Planner**: Schedule optimization and task management
- **Advisor**: Personalized learning advice based on your profile and academic data
- **Profile Management**: Create and manage student profiles with learning styles, academic goals, and performance metrics.
- **AI-Enhanced Learning Assistant**: Chat with an AI assistant trained on educational resources and personalized to your profile.
- **Notewriter Agent**: Transform content from various sources (text, web pages, PDFs, YouTube videos) into structured study notes tailored to your learning style.
- **Study Planner**: Create and manage study plans with time blocks and task prioritization.
- **Performance Tracking**: Record and analyze academic performance over time.
- **Resource Management**: Save and organize learning resources and references.
- **Advisor**: Receive personalized learning strategies and time management advice.
- **Chatbot Interface**: Ask questions about your academic journey or get assistance with specific topics.

## Requirements

- Python 3.8+
- PostgreSQL 12+
- Groq API key (for all LLM capabilities - this project does not use OpenAI)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Academic-AI-Assistant.git
cd Academic-AI-Assistant
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL:
   - Install PostgreSQL if you haven't already
   - Update the connection settings in the `.env` file if needed
   - Run the database initialization script:
   ```bash
   python init_db.py
   ```

4. Create `.env` file with your configurations:

```
# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=academic_assistant
DB_USER=postgres
DB_PASSWORD=postgres

# LLM API Keys
GROQ_API_KEY=your_groq_api_key

# Application Settings
DEBUG=True
SECRET_KEY=your_secret_key_here
```

## Usage

Run the Streamlit application:

```bash
streamlit run academic_ai_assistant.py
```

The application will be available at `http://localhost:8501`.

## Agent Architecture

The Academic AI Assistant uses a multi-agent architecture:

1. **Coordinator Agent**: Orchestrates the interaction between specialized agents and manages the overall system state
2. **Planner Agent**: Handles calendar integration and schedule optimization
3. **Notewriter Agent**: Processes academic content and generates study materials
4. **Advisor Agent**: Provides personalized learning and time management advice

These agents work together through LangGraph workflows to provide a comprehensive academic support system.

## Database Structure

The application uses PostgreSQL to store:

- Student profiles
- Tasks and deadlines
- Notes and study materials
- Knowledge base content

## Development

### Project Structure

```
Academic-AI-Assistant/
├── academic_ai_assistant.py       # Main Streamlit application
├── requirements.txt               # Dependencies
├── .env                          # Environment variables
├── src/
│   ├── __init__.py
│   ├── LLM.py                    # LLM integration
│   ├── data_manager.py           # Data handling
│   └── agents/
│       ├── __init__.py
│       ├── coordinator.py        # Coordinator agent
│       ├── planner.py            # Planner agent
│       ├── notewriter.py         # Notewriter agent
│       └── advisor.py            # Advisor agent
```

### Adding New Features

To add new features:

1. Develop the functionality in the appropriate agent module
2. Update the corresponding Streamlit page in `academic_ai_assistant.py`
3. Update any necessary database tables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds upon the ATLAS concept (Academic Task and Learning Agent System)
- LangGraph for the agent workflow framework
- Streamlit for the web interface

## LLM Integration

The application exclusively uses Groq's powerful LLMs (via chatgroq) to power all AI features:

1. **Chat Interface**: General academic assistance on the home page
2. **Content Processing**: Convert lecture notes and readings into structured study materials
3. **Personalized Advice**: Generate tailored studying advice based on learning style and profile

You'll need a Groq API key to use these features. The application is designed to work solely with Groq's models and does not require any other LLM provider.