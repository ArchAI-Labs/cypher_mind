# Use a base Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements_docker.txt

# Expose port 7687 (Neo4j Bolt port)
EXPOSE 7687

# Command to run the Python application
CMD ["python", "src/main.py"]