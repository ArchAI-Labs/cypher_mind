# Multi-stage build for optimal image size
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_docker.txt .

# Install dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements_docker.txt

# Install spaCy model
RUN /venv/bin/python -m spacy download en_core_web_sm

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv

# Set environment variables
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy application code
COPY --chown=app:app . /app

# Switch to non-root user
USER app

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