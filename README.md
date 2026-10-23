# Transformix AI Magazine Bot

Telegram bot that collects AI and tech news (NewsAPI, GNews, RSS: VentureBeat AI, TechCrunch AI,
OpenAI blog), extracts the full articles, and uses Claude on AWS Bedrock to write daily summaries,
weekly and monthly blogs, PDF reports, and the monthly **Transformix AI Report** magazine.

Production: VPS4 (`188.166.16.251`), `/home/ubuntu/ai_news`, container `ai-news-bot-container`.

## Magazine design

The magazine layout is ported from the JTGC premium magazine (members.jtgc.sa) and re-skinned in
the Transformix brand (ink `#131316`, violet `#7c5cff`, Sora + Source Serif 4).

| Page | Content |
|---|---|
| Cover | Full-bleed photo, Transformix mark and masthead, issue chip, headline, three highlights |
| Editor's note | Hero photo with the note on a floating card |
| In this issue | 2-column cards with thumbnails and live page numbers |
| Articles (8) | One column under a full-width photo, category pill, source and intelligence score |
| Market watch | The month's AI economy indicators, plus the sources used in the issue |
| Back cover | Full-bleed photo, mark, ai.thetransformix.com |

Every page carries a real photo. For each story the renderer tries the feed image, then the
article page's `og:image`, then Unsplash (only if `UNSPLASH_ACCESS_KEY` is set), then the bundled
AI imagery in `templates/images`. Each article's hero photo is sized by measuring where its text
ends, so pages end full without overflowing. Text is cut only when the photo is already at its
minimum height.

| File | Role |
|---|---|
| `telegram_bot_v2.py` | Bot, news fetching, prompts (`generate_magazine_content_with_ai`) |
| `magazine.py` | Image sourcing, text fitting, WeasyPrint rendering |
| `templates/magazine.html`, `magazine.css` | Layout |
| `templates/fonts/` | Bundled Sora and Source Serif 4 (OFL), so rendering needs no network |
| `render_magazine.py` | Re-render the last issue from saved JSON without a new AI generation |

Each `/magazine` run saves its content to `weekly_blogs/last_magazine.json`. To try a layout
change without paying for a new generation:

```bash
docker exec ai-news-bot-container python render_magazine.py weekly_blogs/last_magazine.json /tmp/mag.pdf
```

## Run

```bash
cp .env.example .env   # fill in the keys
docker compose up -d --build
docker compose logs -f
```

Telegram commands: `/start`, `/news`, `/categories`, `/weekly`, `/monthly`, `/magazine`, `/keywords`.

## Deploy to VPS4

The container has no code bind mount, so a code change needs a rebuild:

```bash
cd /home/ubuntu/ai_news
docker compose build && docker compose up -d
docker exec ai-news-bot-container md5sum magazine.py templates/magazine.html   # compare with the host copy
```

`.env` is copied into the image (`COPY . .`) and read by `load_dotenv()`, which is how the Bedrock
variables reach the bot. `docker-compose.yml` only forwards the four Telegram/news keys.
