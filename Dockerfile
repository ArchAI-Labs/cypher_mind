# Use a base Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Aggiorna i pacchetti e installa git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements_docker.txt

# Install spaCy English model
RUN python -m spacy download en_core_web_sm

# Verify installations
RUN python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model ready')" && \
    python -c "import fastembed; print('✅ FastEmbed ready')" && \
    python -c "from qdrant_client import QdrantClient; print('✅ Qdrant client ready')"

# Expose port
EXPOSE 7687

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import spacy; spacy.load('en_core_web_sm')" || exit 1

# Run application
CMD ["python", "src/main.py"]