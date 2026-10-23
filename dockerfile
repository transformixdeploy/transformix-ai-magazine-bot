FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for newspaper3k and other libraries
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libjpeg-dev \
    libopenjp2-7-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "telegram_bot_v2.py"]