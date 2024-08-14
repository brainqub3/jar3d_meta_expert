# Dockerfile for Jar3d
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal required build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Chainlit runs on
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "chat.py", "--port", "8000"]