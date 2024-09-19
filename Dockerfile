# Dockerfile for Jar3d
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal required build tools and dependencies for Playwright
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its dependencies
# RUN playwright install-deps
# RUN playwright install chromium firefox webkit

# Copy the rest of the application
COPY . .

# Ensure the config file is copied to the correct location
# COPY config/config.yaml /app/config/config.yaml
COPY config/config.yaml /config/config.yaml
COPY agent_memory/jar3d_final_response_previous_run.txt /app/agent_memory/jar3d_final_response_previous_run.txt


# Expose the port Chainlit runs on
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "chat.py", "--port", "8000"]