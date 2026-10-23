import requests
import json
from datetime import datetime, timedelta
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import os
import tempfile

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without loading .env file
    pass
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time
from newspaper import Article
import nltk
from readability import readability
import feedparser
from jinja2 import Environment, FileSystemLoader
import boto3
from botocore.config import Config
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except OSError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint (GTK) not found. PDF generation will be disabled.")
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint module not found.")
# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Your Telegram Bot Token (you'll need to get this from @BotFather)
# Read from the environment (.env); no secrets are kept in source
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# API Keys - Get from environment variables for production security
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# AWS Bedrock Configuration
# Support both variable name formats for flexibility
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_BEDROCK_REGION", "us-east-1")
AWS_BEDROCK_INFERENCE_PROFILE = os.getenv("AWS_BEDROCK_INFERENCE_PROFILE") or os.getenv("AWS_BEDROCK_INFERENCE_PROFILE_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Initialize AWS Bedrock client
try:
    if not AWS_BEARER_TOKEN_BEDROCK:
        logger.warning("⚠️ AWS_BEARER_TOKEN_BEDROCK environment variable is not set!")
        logger.warning("   Please set it using:")
        logger.warning("   Windows: set AWS_BEARER_TOKEN_BEDROCK=your_token")
        logger.warning("   Linux/Mac: export AWS_BEARER_TOKEN_BEDROCK=your_token")
        raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable is required")
    
    # Standard client with default timeout (60 seconds)
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    
    # Long-running client for magazine generation (600 seconds timeout)
    bedrock_config_long = Config(
        read_timeout=600,
        connect_timeout=10,
        retries={'max_attempts': 1}
    )
    bedrock_client_long = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        config=bedrock_config_long
    )
    
    logger.info(f"✅ AWS Bedrock client initialized successfully")
    logger.info(f"   Region: {AWS_REGION}")
    logger.info(f"   Inference Profile: {AWS_BEDROCK_INFERENCE_PROFILE}")
    logger.info(f"   Standard timeout: 60s, Long operations timeout: 600s")
except Exception as e:
    logger.error(f"❌ Failed to initialize AWS Bedrock client: {str(e)}")
    raise

# RSS feeds for AI news coverage
RSS_FEEDS = [
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
    {"name": "TechCrunch AI", "url": "https://techcrunch.com/tag/artificial-intelligence/feed/"},
    {"name": "OpenAI Blog", "url": "https://openai.com/news/rss.xml"}
]

KEYWORD_INPUT_INSTRUCTIONS = (
    "✍️ *Keyword Setup*\n"
    "Send your keywords in this format:\n"
    "`Primary Keyword | secondary keyword 1, secondary keyword 2, secondary keyword 3`\n\n"
    "Example:\n"
    "`AI News Trends 2025 | generative AI, deep learning, enterprise AI`\n\n"
    "Send *cancel* to exit keyword entry."
)


def parse_keyword_input(raw_text):
    """Parse user provided keyword string into primary and secondary lists."""
    if not raw_text:
        return None
    parts = raw_text.split('|', 1)
    primary = parts[0].strip()
    if not primary:
        return None
    secondary = []
    if len(parts) > 1:
        secondary = [kw.strip() for kw in parts[1].split(',') if kw.strip()]
    return {
        "primary": primary,
        "secondary": secondary
    }


def format_secondary_keywords(secondary_list):
    if not secondary_list:
        return "No secondary keywords specified"
    return ", ".join(secondary_list)


def build_keyword_instruction_block(keywords):
    """Return the SEO/keyword instructions block for prompts."""
    if keywords and keywords.get("primary"):
        primary = keywords["primary"]
        secondary_text = format_secondary_keywords(keywords.get("secondary", []))
        keyword_header = (
            f'PRIMARY KEYWORD: "{primary}"\n'
            f"SECONDARY KEYWORDS / LSI: {secondary_text}\n"
        )
    else:
        keyword_header = (
            "PRIMARY KEYWORD: Not specified (infer the best fit from the article set)\n"
            "SECONDARY KEYWORDS / LSI: Use relevant synonyms and supporting subtopics\n"
        )

    return f"""
{keyword_header}
SEO requirements:
- Place the primary keyword in:
  • The SEO Title
  • The H1
  • The first paragraph (within the first 100 words)
  • Naturally 2–3 times every ~300 words throughout the body (no rigid density; keep it organic)
- Spread secondary/LSI keywords across selected H2/H3 headings and various paragraphs as thematic synonyms.
- Do NOT repeat the exact same keyword in every heading—use natural variations to avoid keyword stuffing.

Mandatory SEO outputs at the top of the response (before any other sections):
1. SEO Title: < 60 characters, includes the primary keyword and highlights a clear benefit.
2. Meta Description: 120–150 characters summarizing the core value, optionally includes the primary keyword once (only if it reads naturally) plus a light CTA.
3. Recommended Slug: lowercase, hyphen-separated version of the primary keyword (e.g., ai-trends-2025).
4. Headings Structure: Proposed H2/H3 outline derived from the primary + secondary keywords using varied phrasing.

After providing these SEO elements, continue with the requested blog structure while following the keyword guidance above.
""".strip()


def keywords_summary_text(keywords):
    if not keywords or not keywords.get("primary"):
        return "No keywords have been configured yet."
    secondary = format_secondary_keywords(keywords.get("secondary", []))
    return f"Primary: {keywords['primary']}\nSecondary: {secondary}"


def get_user_keywords(context):
    """Safely fetch stored keywords from user data."""
    try:
        return context.user_data.get("blog_keywords")
    except Exception:
        return None

async def get_news(update: Update, context: ContextTypes.DEFAULT_TYPE, page=1, category=None):
    """Get today's enhanced AI news with presenter-style summary."""
    # Send initial message
    if update.callback_query:
        await update.callback_query.answer()
        message = await update.callback_query.message.reply_text(
            "📺 Preparing your AI news briefing...\n📖 Gathering comprehensive coverage...\nPlease wait, this may take a moment.",
            parse_mode='Markdown'
        )
    else:
        message = await update.message.reply_text(
            "📺 Preparing your AI news briefing...\n📖 Gathering comprehensive coverage...\nPlease wait, this may take a moment.",
            parse_mode='Markdown'
        )
    
    try:
        # Update progress message
        await message.edit_text(
            "📺 *Step 1/3:* Fetching news from multiple sources...",
            parse_mode='Markdown'
        )
        
        #  Fetch news from APIs and RSS feeds
        newsapi_articles = fetch_ai_news() or []  # Ensure it's not None
        gnews_articles = fetch_gnews_ai() or []  # Ensure it's not None
        rss_articles = fetch_rss_ai() or []
        logger.info(f"Fetched {len(newsapi_articles)} NewsAPI articles, {len(gnews_articles)} GNews articles, {len(rss_articles)} RSS articles")
        # Filter relevant articles
        filtered_newsapi = filter_relevant_articles(newsapi_articles) or []
        filtered_gnews = filter_relevant_articles(gnews_articles) or []
        filtered_rss = filter_relevant_articles(rss_articles) or []
        
        # Daily scope: restrict to past 1 day
        recent_rss = filter_recent_articles(filtered_rss, days=2) or []
        recent_gnews = filter_recent_articles(filtered_gnews, days=2) or []
        recent_newsapi = filter_recent_articles(filtered_newsapi, days=2) or []
        
        await message.edit_text(
            "📺 *Step 2/3:* Extracting full article content for analysis...\n📖 This may take 30-60 seconds...",
            parse_mode='Markdown'
        )
        # Enhance articles with full content (daily)
        enhanced_newsapi = enhance_articles_with_content(recent_newsapi, max_articles=20) or []
        enhanced_gnews = enhance_articles_with_content(recent_gnews, max_articles=20) or []
        enhanced_rss = enhance_articles_with_content(recent_rss, max_articles=50) or []
        all_enhanced_articles = enhanced_newsapi + enhanced_gnews + enhanced_rss
        with open("all_enhanced_articles.txt", "w", encoding="utf-8") as f:
            json.dump(all_enhanced_articles, f, ensure_ascii=False, indent=2)
        await message.edit_text(
            "📺 *Step 3/3:* Finalizing your news briefing...",
            parse_mode='Markdown'
        )
        
        # Format the message
        news_message, total_pages, current_category, relevant_articles = format_news_message(
            enhanced_newsapi, enhanced_gnews, enhanced_rss,page, category
        )
        
        # Update message header for presenter style
        if category:
            news_message = f"📺 *AI NEWS BRIEFING - {category.upper()}*\n" + news_message[news_message.find('\n')+1:]
        else:
            news_message = f"📺 *DAILY AI NEWS BRIEFING*\n" + news_message[news_message.find('\n')+1:]
        
        # Create keyboard based on context
        keyboard = []
        
        if category:
            # Category view with pagination
            if total_pages > 1:
                nav_row = []
                if page > 1:
                    nav_row.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'category_{category}_{page-1}'))
                if page < total_pages:
                    nav_row.append(InlineKeyboardButton("Next ➡️", callback_data=f'category_{category}_{page+1}'))
                if nav_row:
                    keyboard.append(nav_row)
            
            # Add PDF download button for category
            keyboard.append([InlineKeyboardButton("📄 Download News Briefing", callback_data=f'pdf_{category}')])
            keyboard.extend([
                [InlineKeyboardButton("🔄 Fresh Update", callback_data=f'category_{category}_1')],
                [InlineKeyboardButton("📊 Other Briefings", callback_data='show_categories')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='main_menu')]
            ])
        else:
            # Main view
            keyboard = [
                [InlineKeyboardButton("📄 Download Complete Briefing", callback_data='pdf_all')],
                [InlineKeyboardButton("🔄 Fresh Update", callback_data='get_news')],
                [InlineKeyboardButton("📝 Generate Weekly Blogs", callback_data='generate_weekly')],
                [InlineKeyboardButton("📊 Specialized Briefings", callback_data='show_categories')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='main_menu')]
            ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Update the message with final results
        await message.edit_text(
            news_message,
            parse_mode='Markdown',
            reply_markup=reply_markup,
            disable_web_page_preview=True
        )
            
    except Exception as e:
        error_message = f"❌ Error preparing news briefing: {str(e)}"
        logger.error(f"News briefing error: {str(e)}")
        await message.edit_text(error_message)

async def show_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show category selection menu."""
    categories_message = """
🤖 *ENHANCED AI NEWS CATEGORIES*

Choose a category to explore with full content extraction:

• 🤖 **LLM & Language Models** - ChatGPT, GPT, Gemini, Claude, etc.
• 💼 **AI Business & Investment** - Startups, funding, market news
• 🔬 **AI Technology & Research** - Research papers, algorithms, models
• ⚖️ **AI Ethics & Regulation** - Policy, safety, responsible AI
• 🛠️ **AI Applications** - Use cases, implementations, products

*🆕 Enhanced Features for Each Category:*
📖 Full article content extraction
🧠 Category-specific AI summaries
📄 Detailed PDF reports with complete content
📊 Content extraction statistics
🔍 Smart filtering based on full text

*Content Quality:*
Each category now analyzes complete articles rather than just headlines and descriptions, providing much deeper insights and more accurate summaries.
    """
    
    keyboard = [
        [InlineKeyboardButton("🤖 LLM & Language Models", callback_data='category_LLM & Language Models_1')],
        [InlineKeyboardButton("💼 AI Business & Investment", callback_data='category_AI Business & Investment_1')],
        [InlineKeyboardButton("🔬 AI Technology & Research", callback_data='category_AI Technology & Research_1')],
        [InlineKeyboardButton("⚖️ AI Ethics & Regulation", callback_data='category_AI Ethics & Regulation_1')],
        [InlineKeyboardButton("🛠️ AI Applications", callback_data='category_AI Applications_1')],
        [InlineKeyboardButton("🏠 Main Menu", callback_data='main_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Always send a new message instead of editing
    if update.callback_query:
        await update.callback_query.message.reply_text(
            categories_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            categories_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

async def generate_weekly_blogs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate comprehensive weekly AI blog posts."""
    # Send initial message
    if update.callback_query:
        await update.callback_query.answer()
        message = await update.callback_query.message.reply_text(
            "📝 *WEEKLY AI BLOG GENERATOR*\n\n⏳ Preparing comprehensive weekly analysis...\n📊 This will analyze 50+ articles from the past week\n⏰ Expected time: 3-5 minutes\n\nPlease wait...",
            parse_mode='Markdown'
        )
    else:
        message = await update.message.reply_text(
            "📝 *WEEKLY AI BLOG GENERATOR*\n\n⏳ Preparing comprehensive weekly analysis...\n📊 This will analyze 50+ articles from the past week\n⏰ Expected time: 3-5 minutes\n\nPlease wait...",
            parse_mode='Markdown'
        )
    
    try:
        # Step 1: Fetch weekly news
        await message.edit_text(
            "📝 *Step 1/6:* Fetching weekly AI news...\n📡 Gathering articles from the past 7 days...",
            parse_mode='Markdown'
        )
        
        newsapi_articles = fetch_weekly_ai_news() or []  # Ensure not None
        gnews_articles = fetch_weekly_gnews_ai() or []  # Ensure not None
        rss_articles = fetch_rss_ai() or []
        
        logger.info(f"Fetched {len(newsapi_articles)} NewsAPI articles, {len(gnews_articles)} GNews articles, {len(rss_articles)} RSS articles")
        
        # Step 2: Filter and combine articles
        await message.edit_text(
            "📝 *Step 2/6:* Filtering relevant articles...\n🔍 Applying AI-specific filters...",
            parse_mode='Markdown'
        )
        
        filtered_newsapi = filter_relevant_articles(newsapi_articles) or []
        filtered_gnews = filter_relevant_articles(gnews_articles) or []
        filtered_rss = filter_relevant_articles(rss_articles) or []
        # Filter for recent articles (past week)
        recent_newsapi = filter_recent_articles(filtered_newsapi, days=7) or []
        recent_gnews = filter_recent_articles(filtered_gnews, days=7) or []
        recent_rss = filter_recent_articles(filtered_rss, days=7) or []
        
        all_articles = recent_newsapi + recent_gnews + recent_rss
        logger.info(f"Total relevant articles: {len(all_articles)}")
        
        if not all_articles:
            await message.edit_text(
                "❌ No recent AI articles found. Please try again later.",
                parse_mode='Markdown'
            )
            return
        
        await message.edit_text(
            f"📝 *Step 3/6:* Extracting full content...\n📖 Processing {min(len(all_articles), 50)} articles\n⏱️ This may take 2-3 minutes...",
            parse_mode='Markdown'
        )
        
        # Step 3: Enhance articles with full content
        enhanced_articles = enhance_articles_with_content(all_articles, max_articles=50, weekly_mode=True) or []
        
        enhanced_count = len([a for a in enhanced_articles if a.get('full_content')])
        logger.info(f"Enhanced articles: {enhanced_count}/{len(enhanced_articles)}")
        
        # Step 4: Categorize articles for blogs
        await message.edit_text(
            "📝 *Step 4/6:* Categorizing articles for blog themes...\n🎯 Technical vs Business focus...",
            parse_mode='Markdown'
        )
        
        categorized = categorize_articles_for_blogs(enhanced_articles)
        
        technical_articles = categorized.get('technical', []) or []
        business_articles = categorized.get('business', []) or []
        
        logger.info(f"Technical blog articles: {len(technical_articles)}, Business blog articles: {len(business_articles)}")
        
        # Check if we have enough articles
        if not technical_articles and not business_articles:
            await message.edit_text(
                "❌ Unable to categorize articles for blog generation. Please try again later.",
                parse_mode='Markdown'
            )
            return
        
        user_keywords = get_user_keywords(context)
        
        # Step 5: Generate blog posts
        await message.edit_text(
            "📝 *Step 5/6:* Generating blog posts with AI...\n✍️ Creating comprehensive weekly analysis...",
            parse_mode='Markdown'
        )
        
        # Generate Technical Blog
        technical_blog = None
        if technical_articles:
            technical_blog = generate_blog_post_with_ai(
                technical_articles, "technical", keywords=user_keywords
            )
        
        # Generate Business Blog
        business_blog = None
        if business_articles:
            business_blog = generate_blog_post_with_ai(
                business_articles, "business", keywords=user_keywords
            )
        
        # Step 6: Create PDFs
        await message.edit_text(
            "📝 *Step 6/6:* Creating professional PDF documents...\n📄 Formatting blog posts...",
            parse_mode='Markdown'
        )
        
        # Create PDFs with improved blog formatting
        technical_filename = None
        business_filename = None
        
        if technical_blog:
            technical_filename = create_blog_pdf(
                technical_blog, 
                "AI Technology & Research Weekly",
                is_temp_file=True
            )
        
        if business_blog:
            business_filename = create_blog_pdf(
                business_blog,
                "AI Business & Industry Weekly", 
                is_temp_file=True
            )
        
        # Send the blog PDFs
        if technical_filename:
            try:
                with open(technical_filename, 'rb') as pdf_file:
                    await message.reply_document(
                        document=pdf_file,
                        filename=f"AI_Tech_Weekly_{datetime.now().strftime('%Y%m%d')}.pdf",
                        caption="📝 **AI Technology & Research Weekly Blog**\n🔬 Comprehensive analysis of technical developments, research breakthroughs, and innovation trends",
                        parse_mode='Markdown'
                    )
                # Clean up
                os.unlink(technical_filename)
            except Exception as e:
                logger.error(f"Error sending technical PDF: {e}")
        
        if business_filename:
            try:
                with open(business_filename, 'rb') as pdf_file:
                    await message.reply_document(
                        document=pdf_file,
                        filename=f"AI_Business_Weekly_{datetime.now().strftime('%Y%m%d')}.pdf",
                        caption="📝 **AI Business & Industry Weekly Blog**\n💼 Comprehensive analysis of market trends, investments, and industry developments",
                        parse_mode='Markdown'
                    )
                # Clean up
                os.unlink(business_filename)
            except Exception as e:
                logger.error(f"Error sending business PDF: {e}")
        
        # Summary message with statistics
        # Calculate percentages safely to avoid division by zero
        success_rate = 0
        if enhanced_articles and len(enhanced_articles) > 0:
            success_rate = (enhanced_count / len(enhanced_articles)) * 100
        
        # Prepare status messages
        tech_status = "Generated" if technical_blog else "Skipped (insufficient data)"
        business_status = "Generated" if business_blog else "Skipped (insufficient data)"
        
        success_message = f"""
 ✅ **Weekly AI Blog Generation Complete!**

 📊 **Processing Statistics:**
 • Total Articles Analyzed: {len(enhanced_articles)}
 • Content Extraction Success: {enhanced_count}/{len(enhanced_articles)} ({success_rate:.1f}%)
 • Technical Blog Articles: {len(technical_articles)}
 • Business Blog Articles: {len(business_articles)}
 • Week Coverage: {(datetime.now() - timedelta(days=7)).strftime('%B %d')} - {datetime.now().strftime('%B %d, %Y')}

 📝 **Generated Blogs:**
 • AI Technology & Research Weekly - {tech_status}
 • AI Business & Industry Weekly - {business_status}

 Both blogs include structured sections, detailed analysis, and professional formatting!
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Generate Fresh Blogs", callback_data='generate_weekly')],
            [InlineKeyboardButton("📰 Daily News", callback_data='get_news')],
            [InlineKeyboardButton("🏠 Main Menu", callback_data='main_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Clean the message to ensure it's safe for Telegram Markdown parsing
        try:
            await message.edit_text(
                success_message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        except Exception as parse_error:
            logger.error(f"Markdown parsing error: {parse_error}")
            # Fallback to plain text if Markdown fails
            await message.edit_text(
                success_message.replace('*', '').replace('_', ''),
                reply_markup=reply_markup
            )
        
    except Exception as e:
        error_message = f"❌ Error generating weekly blogs: {str(e)}"
        logger.error(f"Weekly blog generation error: {str(e)}")
        try:
            await message.edit_text(error_message)
        except Exception as edit_error:
            logger.error(f"Error editing message: {edit_error}")
            # Try to send a new message instead
            try:
                await message.reply_text(error_message)
            except Exception as reply_error:
                logger.error(f"Error sending reply: {reply_error}")

async def generate_monthly_blog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate comprehensive monthly AI blog post."""
    # Send initial message
    if update.callback_query:
        await update.callback_query.answer()
        message = await update.callback_query.message.reply_text(
            "📝 *MONTHLY AI BLOG GENERATOR*\n\n⏳ Preparing comprehensive monthly analysis...\n📊 This will analyze 50+ articles from the past monthly\n⏰ Expected time: 10-15 minutes\n\nPlease wait...",
            parse_mode='Markdown'
        )
    else:
        message = await update.message.reply_text(
            "📝 *MONTHLY AI BLOG GENERATOR*\n\n⏳ Preparing comprehensive monthly analysis...\n📊 This will analyze 50+ articles from the past monthly\n⏰ Expected time: 10-15 minutes\n\nPlease wait...",
            parse_mode='Markdown'
        )
    
    try:
        # Step 1: Fetch monthly news
        await message.edit_text(
            "📝 *Step 1/5:* Fetching monthly AI news...\n📡 Gathering articles from the past 1 month...",
            parse_mode='Markdown'
        )
        
        newsapi_articles = fetch_monthly_ai_news() or []  # Ensure not None
        gnews_articles = fetch_weekly_gnews_ai() or []  # Ensure not None
        rss_articles = fetch_rss_ai() or []
        
        logger.info(f"Fetched {len(newsapi_articles)} NewsAPI articles, {len(gnews_articles)} GNews articles, {len(rss_articles)} RSS articles")
        
        # Step 2: Filter and combine articles
        await message.edit_text(
            "📝 *Step 2/5:* Filtering relevant articles...\n🔍 Applying AI-specific filters...",
            parse_mode='Markdown'
        )
        
        filtered_newsapi = filter_relevant_articles(newsapi_articles) or []
        filtered_gnews = filter_relevant_articles(gnews_articles) or []
        filtered_rss = filter_relevant_articles(rss_articles) or []
        # Filter for recent articles (past week)
        recent_newsapi = filter_recent_articles(filtered_newsapi, days=30) or []
        recent_gnews = filter_recent_articles(filtered_gnews, days=30) or []
        recent_rss = filter_recent_articles(filtered_rss, days=30) or []
        
        all_articles = recent_newsapi + recent_gnews + recent_rss
        logger.info(f"Total relevant articles: {len(all_articles)}")
        
        if not all_articles:
            await message.edit_text(
                "❌ No recent AI articles found. Please try again later.",
                parse_mode='Markdown'
            )
            return
        
        await message.edit_text(
            f"📝 *Step 3/5:* Extracting full content...\n📖 Processing {min(len(all_articles), 50)} articles\n⏱️ This may take 2-3 minutes...",
            parse_mode='Markdown'
        )
        
        # Step 3: Enhance articles with full content
        enhanced_articles = enhance_articles_with_content(all_articles, max_articles=100, monthly_mode=True) or []
        
        enhanced_count = len([a for a in enhanced_articles if a.get('full_content')])
        logger.info(f"Enhanced articles: {enhanced_count}/{len(enhanced_articles)}")
        
        user_keywords = get_user_keywords(context)
        
        # Step 4: Generate blog posts
        await message.edit_text(
            "📝 *Step 4/5:* Generating blog posts with AI...\n✍️ Creating comprehensive monthly analysis...",
            parse_mode='Markdown'
        )
        
        # Generate Monthly Blog
        monthly_blog = generate_monthly_blog_with_ai(
            enhanced_articles, keywords=user_keywords
        )
        
        # Step 5: Create PDFs
        await message.edit_text(
            "📝 *Step 5/5:* Creating professional PDF documents...\n📄 Formatting blog posts...",
            parse_mode='Markdown'
        )
        
        # Create PDFs with improved blog formatting
        monthly_filename = None
        
        if monthly_blog:
            monthly_filename = create_blog_pdf(
                monthly_blog, 
                "AI Technology & Research Monthly",
                is_temp_file=True
            )
        # Send the blog PDFs
        if monthly_filename:
            try:
                with open(monthly_filename, 'rb') as pdf_file:
                    await message.reply_document(
                        document=pdf_file,
                        filename=f"AI_Tech_Monthly_{datetime.now().strftime('%Y%m%d')}.pdf",
                        caption="📝 **AI Technology & Research Monthly Blog**\n🔬 Comprehensive analysis of technical developments, research breakthroughs, and innovation trends",
                        parse_mode='Markdown'
                    )
                # Clean up
                os.unlink(monthly_filename)
            except Exception as e:
                logger.error(f"Error sending monthly PDF: {e}")
        
        # Summary message with statistics
        # Calculate percentages safely to avoid division by zero
        success_rate = 0
        if enhanced_articles and len(enhanced_articles) > 0:
            success_rate = (enhanced_count / len(enhanced_articles)) * 100
        
        # Prepare status messages
        tech_status = "Generated" if monthly_blog else "Skipped (insufficient data)"        
        success_message = f"""
 ✅ **Monthly AI Blog Generation Complete!**

 📊 **Processing Statistics:**
 • Total Articles Analyzed: {len(enhanced_articles)}
 • Content Extraction Success: {enhanced_count}/{len(enhanced_articles)} ({success_rate:.1f}%)
 • Monthly Blog Articles: {len(monthly_blog)}
 • Month Coverage: {(datetime.now() - timedelta(days=30)).strftime('%B %d')} - {datetime.now().strftime('%B %d, %Y')}

 📝 **Generated Blogs:**
 • AI Technology & Research Monthly - {tech_status}

 Monthly blog include structured sections, detailed analysis, and professional formatting!
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Generate Fresh Blogs", callback_data='generate_monthly')],
            [InlineKeyboardButton("📰 Daily News", callback_data='get_news')],
            [InlineKeyboardButton("🏠 Main Menu", callback_data='main_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Clean the message to ensure it's safe for Telegram Markdown parsing
        try:
            await message.edit_text(
                success_message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        except Exception as parse_error:
            logger.error(f"Markdown parsing error: {parse_error}")
            # Fallback to plain text if Markdown fails
            await message.edit_text(
                success_message.replace('*', '').replace('_', ''),
                reply_markup=reply_markup
            )
        
    except Exception as e:
        error_message = f"❌ Error generating monthly blogs: {str(e)}"
        logger.error(f"Monthly blog generation error: {str(e)}")
        try:
            await message.edit_text(error_message)
        except Exception as edit_error:
            logger.error(f"Error editing message: {edit_error}")
            # Try to send a new message instead
            try:
                await message.reply_text(error_message)
            except Exception as reply_error:
                logger.error(f"Error sending reply: {reply_error}")

async def generate_pdf_report(update: Update, context: ContextTypes.DEFAULT_TYPE, category=None):
    """Generate and send enhanced PDF report with full content in weekly blog style."""
    query = update.callback_query
    await query.answer()
    
    # Send status message
    status_message = await query.message.reply_text(
        "📄 Generating enhanced PDF report...\n📖 Extracting full content for detailed analysis...\nPlease wait, this may take 1-2 minutes.",
        parse_mode='Markdown'
    )
    
    try:
        # # Update progress
        # await status_message.edit_text(
        #     "📄 *Step 1/4:* Fetching fresh news data...",
        #     parse_mode='Markdown'
        # )
        
        # Fetch fresh news data
        # newsapi_articles = fetch_ai_news() or []
        # gnews_articles = fetch_gnews_ai() or []
        # rss_articles = fetch_rss_ai() or []
        
        # # Filter relevant articles
        # filtered_newsapi = filter_relevant_articles(newsapi_articles) or []
        # filtered_gnews = filter_relevant_articles(gnews_articles) or []
        # filtered_rss = filter_relevant_articles(rss_articles) or []
        
        # await status_message.edit_text(
        #     "📄 *Step 2/4:* Extracting full article content...",
        #     parse_mode='Markdown'
        # )
        # # Enhance with full content
        # enhanced_newsapi = enhance_articles_with_content(filtered_newsapi, max_articles=20) or []
        # enhanced_gnews = enhance_articles_with_content(filtered_gnews, max_articles=20) or []
        # enhanced_rss = enhance_articles_with_content(filtered_rss, max_articles=20) or []
        # all_enhanced_articles = enhanced_newsapi + enhanced_gnews + enhanced_rss
        await status_message.edit_text(
            "📄 *Step 1/2:* Generating AI blog-style content...",
            parse_mode='Markdown'
        )
        with open("all_enhanced_articles.txt", "r", encoding="utf-8") as f:
            all__articles = json.load(f) 
        # Prepare articles according to scope
        user_keywords = get_user_keywords(context)
        if category and category != 'all':
            categorized = categorize_articles(all__articles)
            articles_for_report = categorized.get(category, [])
            blog_title = f"AI Daily – {category}"
            blog_content = generate_daily_blog_with_ai(articles_for_report, category, keywords=user_keywords)
        else:
            articles_for_report = all__articles
            blog_title = "AI Daily Report"
            blog_content = generate_daily_blog_with_ai(articles_for_report, None, keywords=user_keywords)

        # Fallback if model returned too-short, empty content, or error message
        if not blog_content or len(blog_content.strip()) < 100 or blog_content.startswith("# AI Daily Report") and "Error" in blog_content:
            logger.warning("Model returned empty/short content or error. Using fallback blog content.")
            blog_content = build_fallback_blog_content(articles_for_report, category)
        
        # Build PDF using weekly blog formatter for consistent look
        pdf_filename = create_blog_pdf(blog_content, blog_title, is_temp_file=True)
        report_title = blog_title
        
        await status_message.edit_text(
            "📄 *Step 2/2:* Creating PDF document...",
            parse_mode='Markdown'
        )
        
        # Send the PDF file
        if pdf_filename and os.path.exists(pdf_filename):
            with open(pdf_filename, 'rb') as pdf_file:
                await query.message.reply_document(
                    document=pdf_file,
                    filename=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    caption=f"📄 *{report_title}*\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n📖 Enhanced with full article content extraction",
                    parse_mode='Markdown'
                )
            
            # Clean up the temporary file
            os.unlink(pdf_filename)
            
            # Update status message
            await status_message.edit_text(
                "✅ Enhanced PDF report generated and sent successfully!\n📖 Includes full article content and detailed AI analysis.",
                parse_mode='Markdown'
            )
        else:
            await status_message.edit_text(
                "❌ Error: PDF file could not be created.",
                parse_mode='Markdown'
            )
        
    except Exception as e:
        try:
            await status_message.edit_text(
                f"❌ Error generating enhanced PDF report: {str(e)}",
                parse_mode='Markdown'
            )
        except Exception as edit_error:
            logger.error(f"Error editing status message: {edit_error}")
            # Try to send a new message instead
            try:
                await status_message.reply_text(
                    f"❌ Error generating enhanced PDF report: {str(e)}"
                )
            except Exception as reply_error:
                logger.error(f"Error sending reply: {reply_error}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    
    if query.data == 'get_news':
        await get_news(update, context)
    elif query.data == 'generate_weekly':
        await generate_weekly_blogs(update, context)
    elif query.data == 'generate_monthly':
        await generate_monthly_blog(update, context)
    elif query.data == 'generate_magazine':
        await generate_magazine(update, context)
    elif query.data == 'show_categories':
        await show_categories(update, context)
    elif query.data == 'help':
        await help_command(update, context)
    elif query.data == 'main_menu':
        await start(update, context)
    elif query.data.startswith('pdf_'):
        # Handle PDF generation
        category = query.data.replace('pdf_', '')
        if category == 'all':
            await generate_pdf_report(update, context, None)
        else:
            await generate_pdf_report(update, context, category)
    elif query.data.startswith('category_'):
        # Handle category navigation
        parts = query.data.split('_')
        if len(parts) >= 3:
            category = '_'.join(parts[1:-1])  # Reconstruct category name
            page = int(parts[-1])
            await get_news(update, context, page, category)

async def keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allow the user to set or clear primary/secondary keywords."""
    message = update.message or update.effective_message
    if not message:
        return
    
    user_input = ''
    if context.args:
        user_input = ' '.join(context.args).strip()
    
    if user_input:
        lowered = user_input.lower()
        if lowered in ('clear', 'reset', 'remove', 'none'):
            context.user_data.pop('blog_keywords', None)
            context.user_data.pop('awaiting_keywords_input', None)
            await message.reply_text("🧹 Keyword preferences cleared. Use /keywords whenever you want to set new ones.")
            return
        
        parsed = parse_keyword_input(user_input)
        if parsed:
            context.user_data['blog_keywords'] = parsed
            context.user_data.pop('awaiting_keywords_input', None)
            await message.reply_text(f"✅ Keywords saved!\n{keywords_summary_text(parsed)}")
        else:
            await message.reply_text(
                "⚠️ I couldn't understand that format. Please use:\n"
                "`Primary Keyword | secondary keyword 1, secondary keyword 2`\n"
                "Example: `AI News Trends 2025 | generative AI, deep learning`",
                parse_mode='Markdown'
            )
        return
    
    context.user_data['awaiting_keywords_input'] = True
    await message.reply_text(KEYWORD_INPUT_INSTRUCTIONS, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages."""
    raw_text = (update.message.text or '').strip()
    if not raw_text:
        return
    
    # Handle pending keyword input
    if context.user_data.get('awaiting_keywords_input'):
        lowered = raw_text.lower()
        if lowered in ('cancel', 'stop', 'skip', 'exit'):
            context.user_data.pop('awaiting_keywords_input', None)
            await update.message.reply_text("Keyword entry cancelled. Use /keywords when you're ready to try again.")
            return
        
        parsed = parse_keyword_input(raw_text)
        if parsed:
            context.user_data['blog_keywords'] = parsed
            context.user_data.pop('awaiting_keywords_input', None)
            await update.message.reply_text(f"✅ Keywords saved!\n{keywords_summary_text(parsed)}")
        else:
            await update.message.reply_text(
                "⚠️ I couldn't understand that format. Please use:\n`Primary Keyword | secondary keyword 1, secondary keyword 2`\n"
                "Example: `AI News Trends 2025 | generative AI, deep learning`",
                parse_mode='Markdown'
            )
        return
    
    # If the user sends a keyword-style message (contains a '|'), treat it as keyword input
    if '|' in raw_text and not raw_text.startswith('/'):
        parsed = parse_keyword_input(raw_text)
        if parsed:
            context.user_data['blog_keywords'] = parsed
            context.user_data.pop('awaiting_keywords_input', None)
            await update.message.reply_text(f"✅ Keywords saved!\n{keywords_summary_text(parsed)}")
            return
        else:
            await update.message.reply_text(
                "⚠️ That looks like a keyword entry, but I couldn't understand the format.\n"
                "Please send it as `Primary Keyword | secondary keyword 1, secondary keyword 2` "
                "or use the /keywords command.",
                parse_mode='Markdown'
            )
            return
    
    text = raw_text.lower()
    
    if any(word in text for word in ['news', 'ai', 'artificial intelligence', 'update', 'enhanced']):
        await get_news(update, context)
    elif any(word in text for word in ['weekly', 'blog', 'blogs', 'generate']):
        await generate_weekly_blogs(update, context)
    elif any(word in text for word in ['monthly', 'blog', 'blogs', 'generate']):
        await generate_monthly_blog(update, context)
    elif any(word in text for word in ['magazine', 'report', 'edition']):
        await generate_magazine(update, context)
    elif any(word in text for word in ['categories', 'category', 'topics']):
        await show_categories(update, context)
    elif any(word in text for word in ['help', 'start', 'menu']):
        await start(update, context)
    else:
        keyboard = [
            [InlineKeyboardButton("📰 Get Enhanced News", callback_data='get_news')],
            [InlineKeyboardButton("📝 Generate Weekly Blogs", callback_data='generate_weekly')],
            [InlineKeyboardButton("📝 Generate Monthly Blogs", callback_data='generate_monthly')],
            [InlineKeyboardButton("📊 Categories", callback_data='show_categories')],
            [InlineKeyboardButton("❓ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 Hi! I'm your Enhanced AI News Bot with full content extraction and weekly blog generation.\n\nChoose an option below or use these commands:\n• /news - Enhanced AI news with full articles\n• /weekly - Generate comprehensive weekly blogs\n• /monthly - Generate the monthly deep-dive blog\n• /keywords - Set your primary & secondary keywords for SEO\n• /categories - Browse by topic\n• /help - More information",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

# ============================================================================
# COMMAND REGISTRATION FOR WEEKLY BLOGS
# ============================================================================

async def weekly_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /weekly command directly."""
    await generate_weekly_blogs(update, context)
async def monthly_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /monthly command directly."""
    await generate_monthly_blog(update, context)


# ============================================================================
# EXISTING DAILY NEWS FUNCTIONS (keeping all your original functionality)
# ============================================================================

def extract_article_content(url, max_retries=3):
    """Extract full article content from URL using multiple methods"""
    if not url or url.strip() == '':
        return None
    
    content = None
    
    # Method 1: Try newspaper3k first (most reliable for news articles)
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if article.text and len(article.text.strip()) > 200:
            content = {
                'text': article.text.strip(),
                'title': article.title or '',
                'authors': article.authors or [],
                'publish_date': article.publish_date,
                'method': 'newspaper3k'
            }
            logger.info(f"Successfully extracted content using newspaper3k for {url}")
            return content
    except Exception as e:
        logger.warning(f"Newspaper3k failed for {url}: {str(e)}")
    
    # Method 2: Manual web scraping with BeautifulSoup
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'ads']):
                element.decompose()
            
            # Try multiple content selectors
            content_selectors = [
                'article',
                '[role="main"]',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.story-body',
                '.article-body',
                '.post-body'
            ]
            
            article_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(strip=True, separator=' ')
                        if len(text) > len(article_text):
                            article_text = text
                    break
            
            # Fallback: extract all paragraphs
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up the text
            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = article_text.strip()
            
            if article_text and len(article_text) > 200:
                content = {
                    'text': article_text,
                    'title': soup.find('title').get_text(strip=True) if soup.find('title') else '',
                    'method': 'beautifulsoup'
                }
                logger.info(f"Successfully extracted content using BeautifulSoup for {url}")
                return content
                
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}) for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error extracting content from {url}: {str(e)}")
            break
    
    logger.error(f"Failed to extract content from {url} after all attempts")
    return None

def enhance_articles_with_content(articles, max_articles=15, weekly_mode=False, monthly_mode=False):
    """Enhance articles with full content extraction"""
    if not articles:
        logger.warning("No articles provided to enhance_articles_with_content")
        return []
    
    enhanced_articles = []
    
    # Adjust for weekly vs daily processing
    if weekly_mode:
        max_articles = min(max_articles, 50)  # More articles for weekly
        delay = 0.3  # Shorter delay for bulk processing
    elif monthly_mode:
        max_articles = min(max_articles, 100)  # More articles for monthly
        delay = 0.5  # Shorter delay for bulk processing
    else:
        max_articles = min(max_articles, 20)  # Standard daily limit
        delay = 0.5  # Standard delay
    
    logger.info(f"Starting content extraction for {min(len(articles), max_articles)} articles")
    
    for i, article in enumerate(articles[:max_articles]):
        try:
            url = article.get('url', '') if article else ''
            if not url:
                continue
                
            logger.info(f"Extracting content {i+1}/{min(len(articles), max_articles)}: {url}")
            
            # Extract full content
            content_data = extract_article_content(url)
            
            # Enhance article with extracted content
            enhanced_article = article.copy()
            if content_data and content_data.get('text'):
                enhanced_article['full_content'] = content_data['text']
                enhanced_article['extraction_method'] = content_data['method']
                enhanced_article['content_length'] = len(content_data['text'])
                
                # Use extracted title if original is missing/short
                extracted_title = content_data.get('title', '')
                original_title = article.get('title', '')
                if extracted_title and original_title and len(extracted_title) > len(original_title):
                    enhanced_article['enhanced_title'] = extracted_title
            else:
                description = article.get('description', 'No content available')
                enhanced_article['full_content'] = description or 'No content available'
                enhanced_article['extraction_method'] = 'fallback'
                enhanced_article['content_length'] = len(description) if description else 0
            
            enhanced_articles.append(enhanced_article)
            
            # Respectful delay
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error processing article {i+1}: {str(e)}")
            # Add original article without enhancement
            enhanced_articles.append(article)
            continue
    
    logger.info(f"Content extraction completed. Enhanced {len([a for a in enhanced_articles if a.get('full_content')])} articles")
    return enhanced_articles

def fetch_ai_news():
    """Fetch AI news from NewsAPI"""
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'artificial intelligence OR AI OR machine learning OR deep learning OR ChatGPT OR OpenAI OR Gemini OR Claude OR LLM',
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWSAPI_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f"Error fetching NewsAPI articles: {e}")
        return []

def fetch_gnews_ai():
    """Fetch AI news from GNews"""
    url = 'https://gnews.io/api/v4/search'
    params = {
        'q': 'artificial intelligence OR AI OR machine learning OR deep learning OR ChatGPT OR OpenAI OR Gemini OR Claude OR LLM',
        'lang': 'en',
        'country': 'us',
        'max': 20,
        'apikey': GNEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f"Error fetching GNews articles: {e}")
        return []

def filter_relevant_articles(articles, extra_keywords=None):
    """Filter articles to keep only AI-relevant content"""
    if not articles:
        logger.warning("No articles provided to filter_relevant_articles")
        return []

    # Core AI-related keywords
    ai_keywords = [
        r"artificial intelligence",
        r"\bai\b",  # ensures standalone "AI"
        r"machine learning", r"\bml\b",
        r"deep learning", r"neural networks?",
        r"chatgpt", r"openai", r"gemini", r"claude", r"\bllms?\b",
        r"generative ai", r"\bgenai\b",
        r"automation", r"robotics?", r"data science",
        r"algorithms?", r"computer vision",
        r"\bnlp\b", r"natural language processing",
        r"transformers?", r"\bgpt\b",
        r"anthropic", r"google ai", r"microsoft ai"
    ]

    # Add user-supplied extra keywords if any
    if extra_keywords:
        ai_keywords.extend([re.escape(k.lower()) for k in extra_keywords])

    # Compile regex pattern for efficiency
    pattern = re.compile(r"|".join(ai_keywords), flags=re.IGNORECASE)

    filtered_articles = []
    for article in articles:
        if not article:
            continue

        # Gather searchable text fields
        text_parts = [
            article.get("title", "") or "",
            article.get("description", "") or "",
            article.get("content", "") or ""
        ]
        combined_text = " ".join(text_parts).lower()

        # Match with regex
        if pattern.search(combined_text):
            filtered_articles.append(article)

    return filtered_articles

def categorize_articles(articles):
    """Categorize articles by AI topics"""
    if not articles:
        logger.warning("No articles provided to categorize_articles")
        return {
            'LLM & Language Models': [],
            'AI Business & Investment': [],
            'AI Technology & Research': [],
            'AI Ethics & Regulation': [],
            'AI Applications': []
        }
    
    categories = {
        'LLM & Language Models': [],
        'AI Business & Investment': [],
        'AI Technology & Research': [],
        'AI Ethics & Regulation': [],
        'AI Applications': []
    }
    
    llm_keywords = ['llm', 'language model', 'chatgpt', 'openai', 'gemini', 'claude', 'gpt', 'transformer', 'bert', 'large language']
    business_keywords = ['investment', 'funding', 'startup', 'company', 'business', 'market', 'revenue', 'profit', 'valuation', 'ipo']
    tech_keywords = ['research', 'algorithm', 'neural network', 'deep learning', 'machine learning', 'model', 'architecture', 'breakthrough']
    ethics_keywords = ['ethics', 'regulation', 'policy', 'safety', 'bias', 'privacy', 'security', 'responsible', 'governance']
    app_keywords = ['application', 'use case', 'implementation', 'deployment', 'product', 'service', 'tool', 'solution']
    
    for article in articles:
        if not article:
            continue
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        full_content = article.get('full_content', '') or ''
        title_lower = title.lower()
        description_lower = description.lower()
        full_content_lower = full_content.lower()
        content = f"{title_lower} {description_lower} {full_content_lower[:500]}"  # Include some full content for better categorization
        
        if any(keyword in content for keyword in llm_keywords):
            categories['LLM & Language Models'].append(article)
        elif any(keyword in content for keyword in business_keywords):
            categories['AI Business & Investment'].append(article)
        elif any(keyword in content for keyword in tech_keywords):
            categories['AI Technology & Research'].append(article)
        elif any(keyword in content for keyword in ethics_keywords):
            categories['AI Ethics & Regulation'].append(article)
        elif any(keyword in content for keyword in app_keywords):
            categories['AI Applications'].append(article)
        else:
            # Default to AI Applications if no specific category matches
            categories['AI Applications'].append(article)
    
    return categories

def call_claude_api(system_message, user_message, api_key=None, model=None, max_tokens=16384, temperature=0.7, use_cache=True, use_long_timeout=False):
    """
    Helper function to call AWS Bedrock Claude API.
    Maintains the same signature as the old API function for compatibility.
    
    Args:
        system_message: The system prompt
        user_message: The user prompt
        api_key: Not used (kept for compatibility), uses global client
        model: Not used (kept for compatibility), uses global inference profile
        max_tokens: Maximum tokens in response (default: 16384)
        temperature: Temperature setting (0.0-1.0)
        use_cache: Not used (kept for compatibility)
        use_long_timeout: If True, use 600s timeout (for long operations like magazine generation)
    
    Returns:
        tuple: (response_text, error_message) - error_message is None if successful
    """
    try:
        # Build messages array for AWS Bedrock Claude
        # Combine system and user message since Bedrock Claude uses a simple format
        # We'll prepend the system message to the user message for compatibility
        combined_content = user_message
        if system_message:
            combined_content = f"{system_message}\n\n{user_message}"
        
        messages = [
            {
                "role": "user",
                "content": combined_content
            }
        ]
        
        # Build request body for Bedrock Claude
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": min(max_tokens, 8192),  # Bedrock Claude max is typically 8192
            "temperature": temperature,
            "messages": messages
        }
        
        # Select client based on timeout requirement
        client_to_use = bedrock_client_long if use_long_timeout else bedrock_client
        timeout_info = "600s (long operation)" if use_long_timeout else "60s (standard)"
        
        logger.info(f"Making AWS Bedrock API call - Inference Profile: {AWS_BEDROCK_INFERENCE_PROFILE}, Timeout: {timeout_info}")
        
        # Call AWS Bedrock API
        response = client_to_use.invoke_model(
            modelId=AWS_BEDROCK_INFERENCE_PROFILE,
            body=json.dumps(body)
        )
        
        logger.info(f"AWS Bedrock API call successful")
        
        # Parse response
        response_body = json.loads(response["body"].read())
        
        # Extract content from response
        if response_body.get("content") and len(response_body["content"]) > 0:
            content_text = response_body["content"][0].get("text", "")
            
            if not content_text:
                return None, "AWS Bedrock returned empty content"
            
            return content_text, None
        else:
            return None, "AWS Bedrock returned no content in response"
            
    except Exception as e:
        # Log the full exception for debugging
        error_type = type(e).__name__
        error_str = str(e)
        logger.error(f"AWS Bedrock API error: {error_type}: {error_str}")
        
        error_msg = error_str
        
        # Try to extract more specific error information from AWS Bedrock exceptions
        if hasattr(e, 'response'):
            try:
                if hasattr(e.response, 'get'):
                    error_data = e.response.get('Error', {})
                    if error_data:
                        error_msg = error_data.get('Message', error_msg)
                        error_code = error_data.get('Code', '')
                        logger.error(f"AWS Bedrock error code: {error_code}")
            except:
                pass
        
        # Check for specific error types
        if "ThrottlingException" in error_str or "Too many tokens" in error_str or "throttling" in error_str.lower():
            logger.error("⚠️ Rate limit exceeded - you've hit your daily token limit")
            logger.error("   Solutions:")
            logger.error("   1. Wait until your quota resets (usually daily)")
            logger.error("   2. Check your AWS Bedrock quotas: https://console.aws.amazon.com/servicequotas/")
            logger.error("   3. Request a quota increase if needed")
            logger.error("   4. Try a different model that has available quota")
            error_msg = f"Rate limit exceeded: {error_msg}"
        elif "ValidationException" in error_str or "validation" in error_str.lower():
            logger.error("⚠️ Validation error - check inference profile configuration")
            logger.error(f"   Inference Profile: {AWS_BEDROCK_INFERENCE_PROFILE}")
            logger.error(f"   Region: {AWS_REGION}")
            error_msg = f"Validation error: {error_msg}"
        elif "401" in error_str or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
            logger.error("⚠️ Authentication error - check AWS Bearer Token configuration")
            logger.error(f"   Region: {AWS_REGION}")
            logger.error(f"   Inference Profile: {AWS_BEDROCK_INFERENCE_PROFILE}")
            logger.error("   Make sure AWS_BEARER_TOKEN_BEDROCK environment variable is set correctly")
            error_msg = f"Authentication error: {error_msg}"
        
        return None, f"AWS Bedrock Error: {error_msg}"

def summarize_with_openrouter(articles, api_key=None, category=None):
    """Generate a news presenter style summary from all articles using AWS Bedrock Claude API"""
    if not articles:
        logger.warning("No articles provided to summarize_with_openrouter")
        return "No articles available for summary"
    
    # Prepare consolidated content from all articles
    news_content = ""
    article_titles = []
    
    for i, article in enumerate(articles[:15], 1):  # Process up to 15 articles
        if not article:
            continue
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source') if article.get('source') else 'Unknown source'
        full_content = article.get('full_content', article.get('description', 'No content'))
        
        article_titles.append(f"• {title} ({source})")
        
        # Limit content length to avoid token limits
        if full_content and len(full_content) > 1500:
            full_content = full_content[:1500] + "..."
        
        news_content += f"Article {i}: {title}\nSource: {source}\nContent: {full_content or 'No content'}\n\n"
    
    # System message (static, will be cached)
    system_message = "You are a professional AI news presenter delivering comprehensive briefings. Create engaging, cohesive narratives that flow naturally like a TV news broadcast."
    
    # Create category-specific news presenter prompts
    if category == "LLM & Language Models":
        user_prompt = f"""
        You are a professional AI news presenter giving today's briefing on Large Language Models and Language AI developments.
        
        Create a cohesive, engaging news summary in the style of a TV news presenter. Structure it like a news segment with:
        
        1. **Opening Headlines** (2-3 sentences introducing the biggest stories)
        2. **Main Story Analysis** (4-5 paragraphs flowing naturally between related topics)
        3. **Technology Spotlight** (2-3 paragraphs on key technical developments)
        4. **Market Impact** (1-2 paragraphs on business/industry implications)
        5. **Looking Ahead** (1-2 sentences on future trends)
        
        Make it sound natural and conversational, like you're speaking directly to viewers. Connect related stories together rather than treating them separately. Use transitions like "Meanwhile...", "In related news...", "This development builds on..."
        
        Base your summary on this comprehensive content from today's articles:
        {news_content}
        
        Don't list individual articles - weave the information together into a flowing narrative that tells the complete story of what's happening in LLM and language AI today.
        """
        
    elif category == "AI Business & Investment":
        user_prompt = f"""
        You are a financial news presenter delivering today's AI business and investment update.
        
        Create an engaging, professional news summary structured as:
        
        1. **Market Headlines** (2-3 sentences on the biggest business moves)
        2. **Investment Spotlight** (4-5 paragraphs covering funding, valuations, and deals)
        3. **Strategic Developments** (2-3 paragraphs on partnerships, acquisitions, market positioning)
        4. **Industry Analysis** (2-3 paragraphs on trends and competitive landscape)
        5. **Economic Outlook** (1-2 sentences on what this means for the AI economy)
        
        Speak directly to business professionals and investors. Use financial terminology appropriately and connect market movements to broader economic trends. Make smooth transitions between related stories.
        
        Base your analysis on today's comprehensive business coverage:
        {news_content}
        
        Deliver this as a cohesive business briefing, not separate article summaries.
        """
        
    elif category == "AI Technology & Research":
        user_prompt = f"""
        You are a science and technology correspondent presenting today's AI research and innovation roundup.
        
        Structure your presentation as:
        
        1. **Research Headlines** (2-3 sentences highlighting breakthrough discoveries)
        2. **Scientific Developments** (4-5 paragraphs exploring new methods, findings, and innovations)
        3. **Technical Deep Dive** (2-3 paragraphs on algorithms, architectures, and methodologies)
        4. **Research Impact** (2-3 paragraphs on practical applications and future research directions)
        5. **Innovation Watch** (1-2 sentences on emerging trends in AI research)
        
        Make complex technical concepts accessible while maintaining scientific accuracy. Connect research developments to show how the field is evolving. Use phrases like "Building on this research...", "Scientists have also discovered...", "This breakthrough opens doors to..."
        
        Draw from today's comprehensive research coverage:
        {news_content}
        
        Present this as a unified science briefing that tells the story of AI research progress today.
        """
        
    elif category == "AI Ethics & Regulation":
        user_prompt = f"""
        You are a policy correspondent delivering today's briefing on AI ethics, safety, and regulation.
        
        Present your report with:
        
        1. **Policy Headlines** (2-3 sentences on major regulatory developments)
        2. **Regulatory Landscape** (4-5 paragraphs covering new policies, guidelines, and government actions)
        3. **Ethics in Focus** (2-3 paragraphs on safety concerns, bias issues, and responsible AI initiatives)
        4. **Industry Response** (2-3 paragraphs on how companies and organizations are adapting)
        5. **Future Implications** (1-2 sentences on the path ahead for AI governance)
        
        Adopt a thoughtful, balanced tone appropriate for discussing complex policy issues. Connect related regulatory developments and show how ethics considerations are shaping the AI landscape.
        
        Base your briefing on today's policy and ethics coverage:
        {news_content}
        
        Deliver this as an authoritative policy roundup that connects the dots between different regulatory and ethical developments.
        """
        
    elif category == "AI Applications":
        user_prompt = f"""
        You are a technology correspondent showcasing today's real-world AI applications and implementations.
        
        Structure your segment as:
        
        1. **Innovation Headlines** (2-3 sentences on the most exciting new applications)
        2. **Practical Solutions** (4-5 paragraphs covering new products, services, and implementations)
        3. **Industry Adoption** (2-3 paragraphs on how different sectors are using AI)
        4. **User Impact** (2-3 paragraphs on how these applications affect everyday people and businesses)
        5. **Innovation Trends** (1-2 sentences on where AI applications are heading)
        
        Focus on practical benefits and real-world impact. Make the technology relatable by explaining how it helps solve actual problems. Use transitions that show how different applications build on each other or serve similar needs.
        
        Draw from today's application and implementation news:
        {news_content}
        
        Present this as an engaging showcase of AI in action, not individual product announcements.
        """
        
    else:
        # General AI news presenter style
        user_prompt = f"""
        You are a lead technology correspondent presenting today's comprehensive AI news briefing.
        
        Create an authoritative, engaging news summary structured as:
        
        1. **Today's AI Headlines** (3-4 sentences covering the most significant developments)
        2. **Technology Frontiers** (5-6 paragraphs weaving together technical breakthroughs, new models, and research advances)
        3. **Business & Market Dynamics** (3-4 paragraphs connecting investment news, company developments, and market trends)
        4. **Industry Impact & Applications** (3-4 paragraphs on real-world implementations and sector adoption)
        5. **Regulatory & Ethics Watch** (2-3 paragraphs on policy developments and responsible AI initiatives)
        6. **Looking Forward** (2-3 sentences on emerging trends and future implications)
        
        Deliver this like a seasoned news anchor - authoritative yet accessible, connecting stories to show the bigger picture of AI's evolution. Use smooth transitions and avoid listing individual articles.
        
        Base your comprehensive briefing on today's full coverage:
        {news_content}
        
        Make this sound like a professional news broadcast that tells the complete story of AI developments today.
        """
    
    # Call AWS Bedrock Claude API
    summary, error = call_claude_api(
        system_message=system_message,
        user_message=user_prompt,
        max_tokens=2500,
        temperature=0.4
    )
    
    if error:
        return error, []
    else:
        return summary, article_titles


def format_news_message(newsapi_articles, gnews_articles, rss_articles,page=1, category=None):
    """Format news for Telegram message with pagination and categories"""
    articles_per_page = 6  # Reduced since we have more detailed content
    all_articles = (newsapi_articles or []) + (gnews_articles or []) + (rss_articles or [])
    
    if category and category in ['LLM & Language Models', 'AI Business & Investment', 'AI Technology & Research', 'AI Ethics & Regulation', 'AI Applications']:
        # Show articles from specific category
        categorized = categorize_articles(all_articles)
        category_articles = categorized.get(category, [])
        total_pages = (len(category_articles) + articles_per_page - 1) // articles_per_page if category_articles else 1
        start_idx = (page - 1) * articles_per_page
        end_idx = start_idx + articles_per_page
        page_articles = category_articles[start_idx:end_idx]
        
        message = f"🤖 *AI NEWS - {category.upper()}* (Enhanced)\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        message += f"📄 Page {page} of {total_pages} | Articles: {len(category_articles)}\n"
        
        # Show content extraction stats
        enhanced_count = len([a for a in category_articles if a and a.get('full_content')])
        message += f"📚 Enhanced with full content: {enhanced_count}/{len(category_articles)}\n\n"
        
        message += f"📰 *Articles (Page {page}):*\n"
        for i, article in enumerate(page_articles, start_idx + 1):
            if not article:
                continue
            title = article.get('title', 'No title')
            source = article.get('source', {}).get('name', 'Unknown') if article.get('source') else 'Unknown'
            url = article.get('url', '')
            extraction_method = article.get('extraction_method', 'N/A')
            content_length = article.get('content_length', 0)
            
            if len(title) > 65:
                title = title[:62] + "..."
            
            message += f"{i}. {title}\n"
            message += f"   🏢 {source} | 🔧 {extraction_method}\n"
            message += f"   📊 Content: {content_length} chars\n"
            if url:
                message += f"   🔗 [Read More]({url})\n"
            message += "\n"
        
        return message, total_pages, category, category_articles
    
    else:
        # Show main summary with top articles
        message = f"🤖 *AI NEWS UPDATE* (Enhanced with Full Content)\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Summary Statistics
        total_articles = len(newsapi_articles or []) + len(gnews_articles or []) + len(rss_articles or [])
        enhanced_count = len([a for a in all_articles if a and a.get('full_content')])
        
        message += f"📊 *Summary Statistics:*\n"
        message += f"• Total Articles: {total_articles}\n"
        message += f"• Enhanced with Full Content: {enhanced_count}\n"
        message += f"• NewsAPI: {len(newsapi_articles or [])}\n"
        message += f"• GNews: {len(gnews_articles or [])}\n"
        message += f"• RSS (VentureBeat, TechCrunch, OpenAI): {len(rss_articles or [])}\n\n"
        
        # Top Articles Preview
        message += f"📰 *Top Enhanced Articles Today:*\n"
        
        # Show top 6 articles total
        top_articles = all_articles[:10]
        for i, article in enumerate(top_articles, 1):
            if not article:
                continue
            title = article.get('title', 'No title')
            source = article.get('source', {}).get('name', 'Unknown') if article.get('source') else 'Unknown'
            url = article.get('url', '')
            extraction_method = article.get('extraction_method', 'N/A')
            content_length = article.get('content_length', 0)
            
            if len(title) > 65:
                title = title[:62] + "..."
            
            message += f"{i}. {title}\n"
            message += f"   🏢 {source} | 🔧 {extraction_method}\n"
            message += f"   📊 Content: {content_length} chars\n"
            if url:
                message += f"   🔗 [Read More]({url})\n"
            message += "\n"
        
        return message, 1, None, all_articles

def fetch_rss_ai():
    """Fetch AI news from configured RSS feeds (VentureBeat AI, TechCrunch AI, OpenAI Blog)."""
    articles = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])  # type: ignore[index]
            entries = parsed.entries if hasattr(parsed, 'entries') else []
            for entry in entries:
                try:
                    title = getattr(entry, 'title', '') or ''
                    summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '') or ''
                    link = getattr(entry, 'link', '') or ''
                    published = getattr(entry, 'published', '') or getattr(entry, 'updated', '') or ''
                    # Normalize date to ISO if possible
                    published_parsed = getattr(entry, 'published_parsed', None) or getattr(entry, 'updated_parsed', None)
                    if published_parsed:
                        try:
                            published_dt = datetime(*published_parsed[:6])
                            published_iso = published_dt.isoformat()
                        except Exception:
                            published_iso = published
                    else:
                        published_iso = published
                    article = {
                        'title': title,
                        'description': summary,
                        'url': link,
                        'publishedAt': published_iso,
                        'source': {'name': feed['name']}
                    }
                    articles.append(article)
                except Exception as inner_e:
                    logger.warning(f"Error parsing RSS entry from {feed['name']}: {inner_e}")
                    continue
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed['name']}: {e}")
            continue
    return articles

def fetch_weekly_ai_news():
    """Fetch AI news from the past week using NewsAPI"""
    # Calculate date range for the past week
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'artificial intelligence OR AI OR machine learning OR deep learning OR ChatGPT OR OpenAI OR Gemini OR Claude OR LLM',
        'sortBy': 'publishedAt',
        'language': 'en',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apiKey': NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f"Error fetching NewsAPI articles: {e}")
        return []

def fetch_weekly_gnews_ai():
    """Fetch weekly AI news from GNews"""
    # GNews doesn't have date range, but we can filter by recent articles
    url = 'https://gnews.io/api/v4/search'
    params = {
        'q': 'artificial intelligence OR AI OR machine learning OR deep learning OR ChatGPT OR OpenAI OR Gemini OR Claude OR LLM',
        'lang': 'en',
        'country': 'us',
        'apikey': GNEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f"Error fetching GNews articles: {e}")
        return []
def fetch_monthly_ai_news():
    """Fetch AI news from the past month using NewsAPI"""
    # Calculate date range for the past month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    print(start_date)
    print(end_date)
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'artificial intelligence OR AI OR machine learning OR deep learning OR ChatGPT OR OpenAI OR Gemini OR Claude OR LLM',
        'sortBy': 'publishedAt',
        'language': 'en',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apiKey': NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f"Error fetching NewsAPI articles: {e}")
        return []

def filter_recent_articles(articles, days=7):
    """Filter articles to only include those from the past week"""
    if not articles:
        return []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_articles = []
    
    for article in articles:
        if not article:
            continue
        published_at = article.get('publishedAt') or article.get('published_at')
        if published_at:
            try:
                # Handle different date formats
                if 'T' in published_at:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    pub_date = datetime.strptime(published_at, '%Y-%m-%d')
                
                if pub_date.replace(tzinfo=None) >= cutoff_date:
                    recent_articles.append(article)
            except Exception as e:
                # If date parsing fails, include the article anyway
                recent_articles.append(article)
        else:
            # If no date available, include the article
            recent_articles.append(article)
    
    return recent_articles

def categorize_articles_for_blogs(articles):
    """Categorize articles into two main blog themes"""
    
    if not articles:
        logger.warning("No articles provided to categorize_articles_for_blogs")
        return {
            'technical': [],
            'business': []
        }
    
    # Blog 1: Technical & Research Focus
    technical_keywords = [
        'research', 'breakthrough', 'model', 'algorithm', 'neural network', 
        'deep learning', 'machine learning', 'architecture', 'training',
        'llm', 'language model', 'transformer', 'gpt', 'gemini', 'claude',
        'performance', 'benchmark', 'dataset', 'open source', 'github'
    ]
    
    # Blog 2: Business & Industry Focus
    business_keywords = [
        'company', 'startup', 'investment', 'funding', 'business', 'market',
        'industry', 'enterprise', 'adoption', 'implementation', 'product',
        'service', 'customer', 'revenue', 'growth', 'partnership',
        'acquisition', 'ipo', 'valuation', 'regulation', 'policy', 'ethics'
    ]
    
    technical_articles = []
    business_articles = []
    general_articles = []
    
    for article in articles:
        if not article:
            continue
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        full_content = article.get('full_content', '') or ''
        title_lower = title.lower()
        description_lower = description.lower()
        full_content_lower = full_content.lower()
        content = f"{title_lower} {description_lower} {full_content_lower[:1000]}"  # First 1000 chars of content
        
        technical_score = sum(1 for keyword in technical_keywords if keyword in content)
        business_score = sum(1 for keyword in business_keywords if keyword in content)
        
        if technical_score > business_score and technical_score > 0:
            technical_articles.append(article)
        elif business_score > 0:
            business_articles.append(article)
        else:
            general_articles.append(article)
    
    # Distribute general articles to balance the blogs
    half_general = len(general_articles) // 2
    technical_articles.extend(general_articles[:half_general])
    business_articles.extend(general_articles[half_general:])
    
    return {
        'technical': technical_articles,
        'business': business_articles
    }

def generate_blog_post_with_ai(articles, blog_type, keywords=None):
    """Generate a blog post using OpenRouter AI based on the prompt structure"""
    
    if not articles:
        logger.warning(f"No articles provided to generate_blog_post_with_ai for {blog_type}")
        return "Unable to generate blog post: No articles provided"
    
    # Prepare content from articles - use more articles for weekly summary
    news_content = ""
    article_count = min(len(articles), 30)  # Increased to 30 articles for truly comprehensive coverage
    
    for i, article in enumerate(articles[:article_count], 1):
        if not article:
            continue
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source') if article.get('source') else 'Unknown source'
        full_content = article.get('full_content', article.get('description', 'No content'))
        published_date = article.get('publishedAt', 'Unknown date')
        url = article.get('url', '')
        
        # For weekly content, include more context but keep it manageable
        if full_content and len(full_content) > 600:
            full_content = full_content[:600] + "..."
        
        news_content += f"""
ARTICLE {i}:
Title: {title}
Source: {source}
Date: {published_date}
URL: {url}
Content: {full_content or 'No content available'}
---
"""
    
    # Create blog-specific prompts based on type
    if blog_type == "technical":
        blog_focus = "AI Technology & Research Breakthroughs"
        blog_angle = """Focus on technical developments, research breakthroughs, new AI models, algorithms, and technological innovations. 
        Your audience consists of AI researchers, developers, and tech enthusiasts who want to understand the latest technical advances.
        Highlight specific technical details, model architectures, performance improvements, and research findings."""
        
    else:  # business
        blog_focus = "AI Business & Industry Impact"
        blog_angle = """Focus on business developments, market trends, industry adoption, investment news, and the commercial impact of AI.
        Your audience consists of business professionals, entrepreneurs, and industry leaders who want to understand AI's business implications.
        Highlight market opportunities, business strategies, investment trends, and real-world applications."""
    
    # System message (static, will be cached)
    system_message = "You are a professional tech blogger who writes engaging, insightful blog posts about AI developments. Create content that is informative yet accessible, with clear structure and actionable insights. Always use proper markdown formatting for headers."
    
    keyword_guidance = build_keyword_instruction_block(keywords)
    
    user_prompt = f"""
    {keyword_guidance}
    
    Write a comprehensive weekly blog post about this week's {blog_focus} with the following EXACT structure.
    
    IMPORTANT: Use these EXACT section headers with proper markdown formatting:

    # [Create an engaging title]

    ## Introduction
    [Write a compelling 150-word introduction that hooks the reader]

    ## This Week's Biggest Story
    [250-300 words covering the most significant development]

    ## Major Breakthrough #2
    [250-300 words on the second biggest story]

    ## Emerging Trends
    [200-250 words on notable trends and patterns]

    ## Industry Spotlight
    [200-250 words focusing on specific companies or sectors]

    ## Quick Hits
    [200-250 words covering 6-8 additional developments in brief]

    ## Market Watch
    [150-200 words on funding, investments, and financial news]

    ## What's Next
    [100-150 words previewing next week's developments]

    ## Final Thoughts
    [Brief concluding paragraph with key takeaways]

    {blog_angle}

    CRITICAL REQUIREMENTS:
    - You MUST use the exact section headers shown above with ## markdown formatting
    - Reference at least 15-20 different articles throughout the blog
    - Include specific company names, numbers, dates, and sources
    - Write in an engaging, professional blog tone
    - Make each section substantive and informative
    - This covers {article_count} articles - use the breadth of information available

    ARTICLE CONTENT TO ANALYZE ({article_count} total articles):
    {news_content}

    Create a comprehensive weekly blog post that serves as the definitive roundup of this week's {blog_focus.lower()}. Write it as a professional blog post that readers would want to bookmark and share.
    """
    
    # Call AWS Bedrock Claude API
    content, error = call_claude_api(
        system_message=system_message,
        user_message=user_prompt,
        max_tokens=3500,
        temperature=0.5
    )
    
    if error:
        return error
    else:
        return content

def generate_monthly_blog_with_ai(articles, keywords=None):
    """Generate a blog post using OpenRouter AI based on the prompt structure"""
    
    if not articles:
        logger.warning(f"No articles provided to generate_blog_post_with_ai")
        return "Unable to generate blog post: No articles provided"
    
    # Prepare content from articles - use more articles for weekly summary
    news_content = ""
    article_count = min(len(articles), 100)  # Increased to 30 articles for truly comprehensive coverage
    
    for i, article in enumerate(articles[:article_count], 1):
        if not article:
            continue
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source') if article.get('source') else 'Unknown source'
        full_content = article.get('full_content', article.get('description', 'No content'))
        published_date = article.get('publishedAt', 'Unknown date')
        url = article.get('url', '')
        
        # For weekly content, include more context but keep it manageable
        if full_content and len(full_content) > 1000:
            full_content = full_content[:1000] + "..."
        
        news_content += f"""
ARTICLE {i}:
Title: {title}
Source: {source}
Date: {published_date}
URL: {url}
Content: {full_content or 'No content available'}
---
"""
    
    # Create blog-specific prompts based on type
    blog_focus = "AI Technology & Research Breakthroughs"
    blog_angle = """Focus on technical developments, research breakthroughs, new AI models, algorithms, and technological innovations. 
    Your audience consists of AI researchers, developers, and tech enthusiasts who want to understand the latest technical advances.
    Highlight specific technical details, model architectures, performance improvements, and research findings."""

    # System message (static, will be cached)
    system_message = "You are a professional tech blogger who writes engaging, insightful blog posts about AI developments. Create content that is informative yet accessible, with clear structure and actionable insights. Always use proper markdown formatting for headers."
    
    keyword_guidance = build_keyword_instruction_block(keywords)
    
    user_prompt = f"""
    {keyword_guidance}
    
    Write a comprehensive **monthly blog post** about this month's {blog_focus} with the following EXACT structure.
    
    IMPORTANT: Use these EXACT section headers with proper markdown formatting:

    # [Create an engaging title]

    ## Introduction
    [Write a compelling 200-word introduction that hooks the reader, sets context for the month]

    ## This Month's Biggest Story
    [350-400 words covering the single most significant development]

    ## Major Breakthrough #2
    [300-350 words on the second most important story]

    ## Emerging Trends
    [300-350 words analyzing notable trends, research themes, or recurring patterns across the month]

    ## Industry Spotlight
    [250-300 words focusing on standout companies, labs, or sectors making impact]

    ## Quick Hits
    [300-400 words covering 8-10 additional developments in brief, 2-3 sentences each]

    ## Market Watch
    [250-300 words on funding, investments, partnerships, and financial news across the month]

    ## Final Thoughts
    [200 words wrapping up with key takeaways and forward-looking insights]

    {blog_angle}

    CRITICAL REQUIREMENTS:
    - You MUST use the exact section headers shown above with ## markdown formatting
    - Reference at least 25–30 different articles throughout the blog
    - Include specific company names, numbers, dates, and sources
    - Write in an engaging, professional blog tone
    - Make each section substantive and informative
    - This covers {article_count} articles - use the breadth of information available

    ARTICLE CONTENT TO ANALYZE ({article_count} total articles):
    {news_content}

    Create a comprehensive **monthly blog post** that serves as the definitive roundup of this month's {blog_focus.lower()}. Write it as a professional blog post that readers would want to bookmark and share.
    """
    
    # Call AWS Bedrock Claude API
    content, error = call_claude_api(
        system_message=system_message,
        user_message=user_prompt,
        max_tokens=5000,
        temperature=0.7
    )
    
    if error:
        return error
    else:
        return content

def generate_daily_blog_with_ai(articles, category=None, keywords=None):
    """Generate a daily blog-style summary so PDFs match weekly blog formatting."""
    if not articles:
        logger.warning("No articles provided to generate_daily_blog_with_ai")
        return "# AI Daily Report\n\nNo articles available today."
    
    # Prepare content from articles (shorter excerpts for daily)
    max_daily_articles = min(len(articles), 40)
    news_content = ""
    for i, article in enumerate(articles[:max_daily_articles], 1):
        if not article:
            continue
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source') if article.get('source') else 'Unknown source'
        full_content = article.get('full_content', article.get('description', 'No content'))
        published_date = article.get('publishedAt', 'Unknown date')
        url = article.get('url', '')
        if full_content and len(full_content) > 450:
            full_content = full_content[:450] + "..."
        news_content += f"""
ARTICLE {i}:
Title: {title}
Source: {source}
Date: {published_date}
URL: {url}
Content: {full_content or 'No content available'}
---
"""
    
    # Choose focus
    if category in [
        "LLM & Language Models",
        "AI Business & Investment",
        "AI Technology & Research",
        "AI Ethics & Regulation",
        "AI Applications"
    ]:
        title_suffix = f" – {category}"
        intro_target = f"today's {category.lower()} developments"
    else:
        title_suffix = ""
        intro_target = "today's most important AI developments"
    
    # System message (static, will be cached)
    system_message = "You write concise, structured daily blog reports about AI news. Keep it professional, readable, and well-organized using Markdown headers."
    
    keyword_guidance = build_keyword_instruction_block(keywords)
    
    user_prompt = f"""
    {keyword_guidance}
    
    Write a concise daily blog-style report about {intro_target} using the EXACT structure below in Markdown. Keep it punchy and informative, suitable for a daily read.

# [Create an engaging title for the day]

## Quick Overview
[80-120 words summarizing the day's key themes and headline developments]

## Top Stories
[2-3 short paragraphs, 80-120 words each, weaving together the most important updates]

## Noteworthy Developments
[Bullet list of 6-8 succinct items, each 1-2 sentences, referencing specific companies, models, or outcomes]

## Market and Impact
[1-2 short paragraphs on business/industry or research impact, depending on content]

## What to Watch Next
[3-5 bullet points on upcoming releases, expected announcements, or emerging trends]

CRITICAL REQUIREMENTS:
- Use the exact section headers above with ## Markdown formatting
- Weave information from many articles; don't simply list them
- Reference concrete names, numbers, and organizations where possible
- Maintain a clear, professional, readable tone suitable for a daily blog

ARTICLES TO ANALYZE ({max_daily_articles} total):
{news_content}
"""
    
    # Call AWS Bedrock Claude API
    content, error = call_claude_api(
        system_message=system_message,
        user_message=user_prompt,
        max_tokens=2200,
        temperature=0.45
    )
    
    if error:
        logger.error(f"Claude API error in daily blog: {error}")
        # Return None to trigger fallback
        return None
    elif not content or len(content.strip()) < 50:
        logger.warning(f"Claude API returned empty or very short content: {len(content) if content else 0} chars")
        # Return None to trigger fallback
        return None
    else:
        # Ensure it has at least an H1 title; if not, prepend one
        if not content.lstrip().startswith('#'):
            prefix_title = f"# AI Daily Report{title_suffix}\n\n"
            return prefix_title + content
        logger.info(f"Model content length: {len(content)}")
        return content

def build_fallback_blog_content(articles, category=None):
    """Build a minimal, readable daily report from available articles when the model response is empty."""
    heading = f"# AI Daily Report – {category}" if category else "# AI Daily Report"
    if not articles:
        return f"{heading}\n\nNo articles available today."
    lines = [heading, "", "## Highlights", ""]
    count = 0
    for art in articles:
        if not art:
            continue
        title = art.get('title') or art.get('headline') or art.get('name')
        desc = art.get('description') or art.get('summary') or art.get('excerpt')
        if not title and not desc:
            continue
        bullet = f"- {title.strip()}" if title else "- (untitled)"
        if desc:
            bullet += f" — {desc.strip()[:240]}"
        lines.append(bullet)
        count += 1
        if count >= 20:
            break
    if count == 0:
        lines.append("- No readable items.")
    lines += ["", "## Notes", "This fallback summary was generated due to an empty model response."]
    return "\n".join(lines)

def parse_blog_sections(blog_content):
    """Parse blog content and return structured sections"""
    if not blog_content:
        logger.warning("No blog content provided to parse_blog_sections")
        return []
    
    sections = []
    current_section = {"title": "", "content": "", "level": 0}
    
    lines = blog_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for headers
        if line.startswith('#'):
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section.copy())
            
            # Start new section
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            
            current_section = {
                "title": title,
                "content": "",
                "level": level
            }
        else:
            # Add to current section content
            current_section["content"] += line + " "
    
    # Don't forget the last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    return sections

def clean_markdown_formatting(text):
    """Clean markdown formatting and prepare for PDF"""
    if not text:
        return ""
    
    # Handle bold text (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    
    # Handle italic text (*text* or _text_)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'<i>\1</i>', text)
    
    # Clean any remaining asterisks or underscores
    text = re.sub(r'[*_]{1,2}', '', text)
    
    # Handle inline code
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Escape any remaining problematic characters for ReportLab
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Restore our intentional HTML tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;font name="Courier"&gt;', '<font name="Courier">').replace('&lt;/font&gt;', '</font>')
    
    return text

def create_blog_pdf(blog_content, blog_title, is_temp_file=True):
    """Create a beautifully formatted blog-style PDF"""
    
    if not blog_content or not blog_title:
        logger.warning("No blog content or title provided to create_blog_pdf")
        return None
    
    if is_temp_file:
        # Create a temporary file for Telegram bot
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        filename = temp_file.name
        temp_file.close()
    else:
        # Create with specific filename for standalone use
        filename = f"{blog_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    doc = SimpleDocTemplate(filename, pagesize=A4, 
                          topMargin=0.75*inch, bottomMargin=0.75*inch,
                          leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    # Define comprehensive blog styles
    styles = getSampleStyleSheet()
    
    # Blog title style (main headline)
    blog_title_style = ParagraphStyle(
        'BlogTitle',
        parent=styles['Heading1'],
        fontSize=28,
        spaceAfter=15,
        spaceBefore=0,
        alignment=TA_CENTER,
        textColor=HexColor('#1a1a1a'),
        fontName='Helvetica-Bold',
        leading=32
    )
    
    # Blog metadata style (date, author info)
    blog_meta_style = ParagraphStyle(
        'BlogMeta',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#666666'),
        fontStyle='italic'
    )
    
    # Section header style (H2)
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=12,
        spaceBefore=24,
        textColor=HexColor('#2c3e50'),
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=0,
        leftIndent=0
    )
    
    # Subsection header style (H3)
    subsection_header_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=16,
        textColor=HexColor('#34495e'),
        fontName='Helvetica-Bold'
    )
    
    # Blog paragraph style
    blog_paragraph_style = ParagraphStyle(
        'BlogParagraph',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        spaceBefore=0,
        alignment=TA_JUSTIFY,
        leading=16,
        textColor=HexColor('#333333'),
        firstLineIndent=0,
        leftIndent=0,
        rightIndent=0
    )
    
    # Intro paragraph style (slightly different)
    intro_style = ParagraphStyle(
        'IntroStyle',
        parent=blog_paragraph_style,
        fontSize=12,
        leading=17,
        textColor=HexColor('#2c3e50'),
        spaceAfter=18
    )
    
    # Highlight/quote style
    highlight_style = ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15,
        spaceBefore=15,
        alignment=TA_LEFT,
        textColor=HexColor('#2980b9'),
        fontName='Helvetica-Bold',
        leftIndent=20,
        rightIndent=20,
        borderWidth=1,
        borderColor=HexColor('#e8f4fd'),
        borderPadding=12,
        backColor=HexColor('#f8fbff')
    )
    
    # Footer style
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=HexColor('#999999'),
        spaceAfter=0,
        spaceBefore=30
    )
    
    # Build the document content
    content = []
    
    # Add blog title
    content.append(Paragraph(blog_title, blog_title_style))
    
    # Add metadata (dynamic for Daily vs Weekly)
    date_str = datetime.now().strftime('%B %d, %Y')
    week_range = f"{(datetime.now() - timedelta(days=7)).strftime('%B %d')} - {datetime.now().strftime('%B %d, %Y')}"
    if 'Daily' in blog_title:
        content.append(Paragraph(f"Daily AI Report • Generated on {date_str}", blog_meta_style))
    elif 'Weekly' in blog_title:
        content.append(Paragraph(f"Weekly AI Report • {week_range} • Generated on {date_str}", blog_meta_style))
    else:
        content.append(Paragraph(f"AI Report • Generated on {date_str}", blog_meta_style))
    
    # Add a subtle separator
    content.append(Spacer(1, 10))
    
    # Parse the blog content into sections
    sections = parse_blog_sections(blog_content)
    
    is_first_section = True
    
    for section in sections:
        title = section['title']
        section_content = section['content'].strip()
        level = section['level']
        
        # Skip empty sections or the main title if it repeats
        if not section_content or (level == 1 and title and blog_title and title.lower() in blog_title.lower()):
            continue
        
        # Add section header based on level
        if level == 1:
            # This would be the main title, already added
            continue
        elif level == 2:
            content.append(Paragraph(title, section_header_style))
        elif level >= 3:
            content.append(Paragraph(title, subsection_header_style))
        
        # Process section content
        paragraphs = [p.strip() for p in section_content.split('.') if p.strip()]
        combined_paragraphs = []
        current_para = ""
        
        # Combine short sentences into proper paragraphs
        for para in paragraphs:
            current_para += para + ". "
            if len(current_para) > 200 or para.endswith(('Inc', 'Corp', 'Ltd', 'AI', 'CEO', 'CTO')):
                combined_paragraphs.append(current_para.strip())
                current_para = ""
        
        if current_para.strip():
            combined_paragraphs.append(current_para.strip())
        
        # Add paragraphs
        for para in combined_paragraphs:
            if len(para) < 50:  # Skip very short fragments
                continue
                
            # Clean up the paragraph
            para = clean_markdown_formatting(para)
            
            # Choose appropriate style
            if title and title.lower() == 'introduction' and is_first_section:
                content.append(Paragraph(para, intro_style))
                is_first_section = False
            elif any(keyword in para.lower() for keyword in ['breaking:', 'important:', 'key takeaway:', 'notable:']):
                content.append(Paragraph(para, highlight_style))
            else:
                content.append(Paragraph(para, blog_paragraph_style))
    
    # Add footer
    content.append(Spacer(1, 30))
    content.append(Paragraph("─" * 50, ParagraphStyle('Separator', parent=footer_style, alignment=TA_CENTER)))
    content.append(Paragraph("🤖 Generated by AI Weekly News Blog System", footer_style))
    content.append(Paragraph(f"© {datetime.now().year} • For educational and informational purposes", footer_style))
    
    # Build PDF
    try:
        doc.build(content)
        logger.info(f"✅ Blog PDF saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"❌ Error creating PDF {filename}: {str(e)}")
        return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = """
🤖 *Welcome to Enhanced AI News Bot!*

I'm your advanced AI news source with full article analysis and weekly blog generation!

*🆕 NEW FEATURES:*
• 📖 Full article content extraction
• 🧠 Enhanced AI summaries based on complete articles
• 📝 Weekly blog generation (Technical & Business focus)
• 🔍 Multiple extraction methods (newspaper3k, BeautifulSoup)
• 📊 Content analysis and statistics

*Available Commands:*
• `/news` - Get today's enhanced AI news
• `/categories` - Browse news by category
• `/weekly` - Generate comprehensive weekly blogs
• `/help` - Show this help message

*Features:*
• 📰 Aggregates from multiple sources
• 📖 Extracts full article content
• 🧠 AI-powered summaries using complete articles
• 📄 Enhanced PDF reports with full content
• 📝 Weekly blog posts in professional format
• 🔍 Intelligent content filtering
• 📊 Detailed extraction statistics
• 🗂️ Smart categorization

Just send `/news` for daily updates or `/weekly` for comprehensive blog posts!
    """
    
    keyboard = [
        [InlineKeyboardButton("📰 Get Enhanced News", callback_data='get_news')],
        [InlineKeyboardButton("📝 Generate Weekly Blogs", callback_data='generate_weekly')],
        [InlineKeyboardButton("📝 Generate Monthly Blogs", callback_data='generate_monthly')],
        [InlineKeyboardButton("🎨 Generate AI Magazine (PDF)", callback_data='generate_magazine')],
        [InlineKeyboardButton("📊 Categories", callback_data='show_categories')],
        [InlineKeyboardButton("❓ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Handle both regular messages and callback queries
    if update.callback_query:
        await update.callback_query.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
🤖 *Enhanced AI News Bot Help*

*🆕 ENHANCED FEATURES:*
• 📖 **Full Article Extraction** - Reads complete articles, not just descriptions
• 🧠 **Smarter Summaries** - AI analysis based on full content
• 📝 **Weekly Blog Generation** - Comprehensive blog posts (Technical & Business)
• 🔍 **Multi-Method Extraction** - Uses newspaper3k + BeautifulSoup
• 📊 **Content Statistics** - Shows extraction success rates

*Commands:*
• `/start` - Welcome message and main menu
• `/news` - Get enhanced AI news with full content
• `/categories` - Browse news by category
• `/weekly` - Generate weekly blog posts
• `/monthly` - Generate the comprehensive monthly blog
• `/keywords` - Set the primary & secondary keywords for SEO
• `/help` - Show this help message

*How Enhanced Extraction Works:*
1. 📡 Fetch news from NewsAPI and GNews
2. 🔍 Extract full article content from URLs
3. 📖 Use newspaper3k and BeautifulSoup methods
4. 🧠 Generate AI summaries from complete articles
5. 📄 Create detailed PDF reports

*Weekly Blog Features:*
• 📝 Two comprehensive blog posts (Technical & Business focus)
• 📊 Analysis of 30+ articles per blog
• 🎯 Professional blog formatting with structured sections
• 📄 Beautiful PDF output with proper typography
• 🔄 Weekly content aggregation and filtering

*Categories Available:*
• 🤖 LLM & Language Models
• 💼 AI Business & Investment  
• 🔬 AI Technology & Research
• ⚖️ AI Ethics & Regulation
• 🛠️ AI Applications

*Benefits of Full Content:*
• More accurate summaries
• Better categorization
• Detailed insights
• Complete context
• Professional blog posts

Use `/news` for daily updates or `/weekly` for comprehensive blogs!
    """
    
    # Handle both regular commands and callback queries
    if update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown')
    else:
        await update.message.reply_text(help_text, parse_mode='Markdown')

# ============================================================================
# NEW AI MAGAZINE FEATURE
# ============================================================================

def generate_magazine_content_with_ai(articles, api_key=None):
    """
    Generate structured JSON content for the monthly magazine using Claude.
    """
    if not articles:
        return None

    # Prepare article context
    articles_context = ""
    for i, article in enumerate(articles[:40]):  # Limit to 40 articles for context
        title = article.get('title', 'No title')
        content = article.get('full_content', '')[:1000] # Truncate for token limits
        articles_context += f"Article {i+1}: {title}\nContent: {content}\n\n"

    system_message = (
        "You are the Editor-in-Chief of 'Transformix AI Report', a premium monthly AI magazine. "
        "Your goal is to sustain a professional, insightful, and visionary tone. "
        "CRITICAL PAGE LAYOUT RULE: EVERY article (including the first one) must fit exactly on ONE A4 page. "
        "NO EXCEPTIONS - ALL 8 articles must be between 350-400 words TOTAL (lead + main content). "
        "STRICT ENFORCEMENT: Count words for EACH article. If any article exceeds 400 words, it will overflow the page. "
        "If any article is below 350 words, it will have excessive whitespace. "
        "Aim for 360-380 words per article for optimal page fill without overflow. "
        "The first article is NOT special - it must follow the same word count rules as all other articles. "
        "Balance depth with brevity - provide comprehensive coverage but stay within the strict 350-400 word limit. "
        "Output ONLY valid JSON matching the specified structure."
    )

    user_prompt = f"""
    Create the content for this month's AI magazine based on these articles:
    {articles_context}

    Return a JSON object with this exact structure (no markdown, just JSON):
    {{
        "title": "AI Monthly: [Catchy Title]",
        "subtitle": "[Engaging Subtitle]",
        "date": "[Current Month Year]",
        "highlights": [
            {{"title": "[Highlight 1]", "description": "[Short description]"}},
            {{"title": "[Highlight 2]", "description": "[Short description]"}},
            {{"title": "[Highlight 3]", "description": "[Short description]"}}
        ],
        "editors_note": "[STRICTLY MAX 200 words. 2 paragraphs of visionary editorial commentary aimed to fit on a single page. Formatted with HTML <p> tags]",
        "market_stats": [
            {{"label": "[Metric 1, e.g. AI Investment]", "value": "[Value, e.g. $2.5B]"}},
            {{"label": "[Metric 2, e.g. New Models]", "value": "[Value]"}},
            {{"label": "[Metric 3]", "value": "[Value]"}}
        ],
        "articles": [
            {{
                "category": "[One of: Strategy, Technology, Ethics, Industry]",
                "title": "[Engaging Magazine Headline]",
                "lead": "[Compelling lead paragraph, 2-3 sentences (approximately 40-60 words) that hook the reader. This word count is included in the 350-400 word total.]",
                "content": "[Main content in HTML format with <h3> subheadings and <p> paragraphs. STRICT WORD COUNT: Total word count (lead + this content) MUST be exactly 350-400 words - NO MORE, NO LESS. The main content should be 290-340 words (lead is 40-60 words). Generate 3-4 well-developed paragraphs (each 60-90 words) plus 1-2 subheadings. DO NOT exceed 400 words total - this causes page overflow (even 2 lines over is unacceptable). DO NOT go below 350 words total - this creates excessive whitespace. Aim for 360-380 words total for optimal page fill without overflow. This applies to ALL articles including the first one - there are NO exceptions. Use proper HTML structure with <h3> for subheadings and <p> for paragraphs.]",
                "source": "[Original Source Name]",
                "score": "[Relevance Score 1-10]"
            }},
            ... (Generate EXACTLY 8 feature articles. DO NOT EXCEED 8.)
        ]
    }}

    IMPORTANT: 
    1. Ensure all double quotes inside string values are properly escaped with backslash (\").
    2. Do not use Markdown line breaks or trailing commas that invalidate JSON.
    3. Output MUST be a single valid JSON string.
    4. STRICT WORD COUNT ENFORCEMENT FOR ALL ARTICLES (NO EXCEPTIONS): 
       - Each article's total word count (lead paragraph + main content) MUST be between 350-400 words.
       - MINIMUM: 350 words (never shorter - creates excessive whitespace).
       - MAXIMUM: 400 words (never longer - causes page overflow - even 2 lines over is unacceptable).
       - OPTIMAL RANGE: 360-380 words per article for perfect page fill without any overflow.
       - The FIRST article follows the SAME rules as all other articles - it is NOT allowed to be longer.
       - Count words carefully for EACH of the 8 articles individually.
       - If any article exceeds 400 words, you MUST condense it to fit within the limit.
       - If any article is below 350 words, you MUST expand it to reach at least 350 words.
       - Consistency is critical - all 8 articles must be similar in length (360-380 words each).
       - This ensures every page fits perfectly on one A4 page without overflow or excessive whitespace.
    """

    logger.info("Calling AWS Bedrock Claude API for magazine content generation...")
    content_text, error = call_claude_api(
        system_message=system_message, 
        user_message=user_prompt, 
        max_tokens=50000,
        use_long_timeout=True  # Use 600 second timeout for magazine generation
    )

    if error:
        logger.error(f"Magazine generation error (AWS Bedrock): {error}")
        logger.error(f"Error type: {type(error)}")
        return None

    try:
        # Clean potential markdown fences
        json_str = content_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode magazine JSON: {e}")
        return None

def render_magazine_pdf(content_data, output_filename="magazine.pdf"):
    """
    Render PDF using Jinja2 and WeasyPrint.
    """
    if not WEASYPRINT_AVAILABLE:
        logger.error("WeasyPrint not available (missing GTK or module). Cannot generate PDF.")
        return None

    try:
        # Setup Jinja2
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('magazine.html')
        
        # Inject default images if available
        import glob
        import random
        import pathlib
        
        images_dir = os.path.join(template_dir, 'images')
        available_images = []
        if os.path.exists(images_dir):
            available_images = (
                glob.glob(os.path.join(images_dir, '*.jpg')) +
                glob.glob(os.path.join(images_dir, '*.png')) +
                glob.glob(os.path.join(images_dir, '*.jpeg')) +
                glob.glob(os.path.join(images_dir, '*.webp'))
            )
        
        # Assign images to articles (round-robin or random)
        if 'articles' in content_data:
            for article in content_data['articles']:
                if available_images:
                    # Convert to file URI safely handling spaces/OS specifics
                    img_path = random.choice(available_images)
                    article['local_image_path'] = pathlib.Path(img_path).as_uri()
        
        # Inject Cover Image and Logo
        cover_path = os.path.join(images_dir, 'cover_generated.png')
        if os.path.exists(cover_path):
            content_data['cover_image_path'] = pathlib.Path(cover_path).as_uri()
            
        logo_path = os.path.join(images_dir, 'TransformiX logo .png')
        if os.path.exists(logo_path):
            content_data['logo_path'] = pathlib.Path(logo_path).as_uri()

        # Inject Back Cover Image
        back_cover_path = os.path.join(images_dir, 'back_cover_generated.png')
        if os.path.exists(back_cover_path):
            content_data['back_cover_path'] = pathlib.Path(back_cover_path).as_uri()

        # Render HTML
        html_out = template.render(**content_data)
        
        # Convert to PDF
        css_path = os.path.join(template_dir, 'magazine.css')
        HTML(string=html_out, base_url=template_dir).write_pdf(
            output_filename, 
            stylesheets=[CSS(css_path)]
        )
        return output_filename
    except Exception as e:
        logger.error(f"PDF rendering error: {e}")
        return None

def clean_deduplicate_articles(articles):
     # Simple helper if not already present
     seen = set()
     clean = []
     for a in articles:
         t = a.get('title')
         if t and t not in seen:
             seen.add(t)
             clean.append(a)
     return clean

async def generate_magazine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /magazine command."""
    # Send initial status
    if update.callback_query:
        await update.callback_query.answer()
        message = await update.callback_query.message.reply_text(
            "🎨 *AI MAGAZINE GENERATOR*\n\n⏳ Commissioning this month's edition...\n🔍 Analyzing monthly trends available...",
            parse_mode='Markdown'
        )
    else:
        message = await update.message.reply_text(
            "🎨 *AI MAGAZINE GENERATOR*\n\n⏳ Commissioning this month's edition...\n🔍 Analyzing monthly trends available...",
            parse_mode='Markdown'
        )

    try:
        # 1. Fetch Monthly News
        await message.edit_text("🎨 *Phase 1/3:* Gathering intelligence...", parse_mode='Markdown')
        
        newsapi_articles = fetch_monthly_ai_news() or []
        gnews_articles = fetch_weekly_gnews_ai() or [] 
        
        all_articles = clean_deduplicate_articles(newsapi_articles + gnews_articles)
        
        if not all_articles:
             await message.edit_text("❌ No sufficient data found for a magazine.")
             return

        # Enhance top articles
        await message.edit_text("🎨 *Phase 2/3:* Curating and enhancing content...", parse_mode='Markdown')
        enhanced_articles = enhance_articles_with_content(all_articles, max_articles=30, monthly_mode=True)

        # 2. Generate Content
        await message.edit_text("🎨 *Phase 3/3:* Designing layout and generating PDF...", parse_mode='Markdown')
        magazine_data = generate_magazine_content_with_ai(enhanced_articles)
        
        if not magazine_data:
             await message.edit_text("❌ Failed to generate magazine content via AI.")
             return

        # 3. Render PDF
        filename = f"Transformix_AI_Magazine_{datetime.now().strftime('%B_%Y')}.pdf"
        pdf_path = render_magazine_pdf(magazine_data, filename)
        
        if pdf_path and os.path.exists(pdf_path):
             await message.reply_document(
                document=open(pdf_path, 'rb'),
                filename=filename,
                caption=f"🎨 **Transformix AI Monthly - {datetime.now().strftime('%B %Y')}**\n\nEnjoy your premium AI report!",
                parse_mode='Markdown'
            )
             # Optional: os.unlink(pdf_path) if running long term
        else:
             await message.edit_text("❌ Failed to render PDF file.")

    except Exception as e:
        logger.error(f"Magazine error: {e}")
        await message.edit_text(f"❌ Error: {str(e)}")

def main():
    """Start the enhanced bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("news", get_news))
    application.add_handler(CommandHandler("categories", show_categories))
    application.add_handler(CommandHandler("weekly", weekly_command))
    application.add_handler(CommandHandler("monthly", monthly_command))
    application.add_handler(CommandHandler("magazine", generate_magazine))
    application.add_handler(CommandHandler("keywords", keywords_command))
    application.add_handler(CommandHandler("setkeywords", keywords_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    print("🤖 Starting Enhanced AI News Bot with Weekly Blog Generation...")
    print("📱 Bot is ready! Send /start to begin.")
    print("✨ Enhanced features:")
    print("   • 📖 Full article content extraction")
    print("   • 🧠 AI summaries based on complete articles")
    print("   • 📝 Weekly blog generation (Technical & Business)")
    print("   • 📄 Enhanced PDF reports with full content")
    print("   • 🔍 Multi-method content extraction")
    print("   • 📊 Content extraction statistics")
    print("   • ⚡ Smarter categorization using full text")
    print("   • 🎯 Professional blog formatting")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()