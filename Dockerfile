# Use official Python image with PostgreSQL client
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE: 1
ENV PYTHONUNBUFFERED: 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["python3", "run.py"]