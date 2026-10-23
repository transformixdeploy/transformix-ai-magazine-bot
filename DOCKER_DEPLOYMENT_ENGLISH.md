# English AI News Bot - Docker Deployment Guide

This guide explains how to deploy the **English AI News Bot** (`telegram_bot_v2.py`) using Docker. This setup is isolated to avoid conflicts with the Arabic bot.

## Directory Structure

Ensure your `ai_news` folder contains:
- `telegram_bot_v2.py` (The bot script)
- `requirements.txt` (Python dependencies)
- `Dockerfile` (Docker configuration)
- `docker-compose.yml` (Service configuration)
- `.env` (Environment variables)

## Deployment Steps

### 1. Upload Files to VPS

Upload the `ai_news` folder to your VPS home directory.

```bash
# Example using SCP (run from your local machine)
scp -r ai_news ubuntu@57.131.31.99:~/
```

### 2. Connect to VPS

SSH into your VPS and navigate to the folder:

```bash
ssh ubuntu@57.131.31.99
cd ~/ai_news
```

### 3. Configure Environment Variables

Create or edit the `.env` file with your API keys:

```bash
nano .env
```

**Required Variables:**
```env
TELEGRAM_TOKEN=your_telegram_bot_token
NEWSAPI_KEY=your_newsapi_key
GNEWS_API_KEY=your_gnews_key
CLAUDE_API_KEY=your_claude_api_key
PYTHONUNBUFFERED=1
```

### 4. Build and Run

Run the following command to build the Docker image and start the container:

```bash
docker-compose up -d --build
```

### 5. Verify Deployment

Check if the container is running:

```bash
docker ps | grep ai-news-bot
```

View the logs:

```bash
docker-compose logs -f
```

## Managing the Bot

- **Stop:** `docker-compose down`
- **Restart:** `docker-compose restart`
- **Logs:** `docker-compose logs -f`
