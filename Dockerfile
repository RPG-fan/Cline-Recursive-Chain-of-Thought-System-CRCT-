FROM python:3.12-slim

WORKDIR /app

# Install git for repo operations
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Create directory structure if needed
RUN mkdir -p src docs strategy_tasks

# Environment variables
ENV PYTHONUNBUFFERED=1

# Create an empty .clinerules file if it doesn't exist
RUN touch .clinerules

# Expose a volume for code editing in VSCode
VOLUME ["/app"]

# Default command that keeps container running for VSCode to connect
CMD ["tail", "-f", "/dev/null"]
