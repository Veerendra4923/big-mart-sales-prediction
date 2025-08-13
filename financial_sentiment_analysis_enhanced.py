# financial_sentiment_analysis_enhanced.py - Complete Sentiment Analysis with Fine-tuned FinBERT and Headlines
import requests
import json
# ... existing code ...
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional, Tuple
import time
import warnings
import os
import logging
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependency: numpy
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    print("⚠️ numpy not installed. Falling back to Python averages.")

# Try to import optional libraries
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("⚠️ feedparser not installed. GNews and RSS feeds will be unavailable.")

try:
    import praw
    HAS_REDDIT = True
except ImportError:
    HAS_REDDIT = False
    print("⚠️ praw not installed. Reddit data will be unavailable.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed. Fine-tuned FinBERT will be unavailable.")

try:
    import yfinance as yf
    import pandas as pd
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ yfinance not installed. Stock market data snapshot will be unavailable.")

# API Configuration - Using environment variables or None
def get_api_key(key_name: str) -> Optional[str]:
    """Safely get API keys from environment or return None"""
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except:
        return os.environ.get(key_name)

API_KEYS = {
               "REDDIT_CLIENT_ID":"Z7R6tVgJpeq1EAhyo7jHEg",
            "REDDIT_CLIENT_SECRET":"dKK2KeYg7JoEUPHlc5nbNGYlTHr5dQ",
            "REDDIT_USER_AGENT": "windows:FINANCIAL_SENTIMENT:1.0 (by u/FinancialAnalyst)",
            "YOUTUBE_API_KEY":"AIzaSyDXf5lMqsQ3DKe6wyCQnG0XEPDvRNCDQEw",
            "TWITTER_BEARER_TOKEN":"AAAAAAAAAAAAAAAAAAAAAJe%2F3AEAAAAA4GluRujTRaSJd8RT1p8F1DNkdh4%3DJ8JlpXujNHqpsDt0eNHqH76NqF6vY6OukWOfUu2RK33cpWye8N",
            "ALPHA_VANTAGE_API_KEY":"Y34KSIIM0WNDVUEW",
            "NEWS_API_KEY": "880e4de1ff6640d7b3a9bd861b983bfd",
}

class FinancialFinBERTAnalyzer:
    """Fine-tuned FinBERT sentiment analyzer trained on financial data"""

    def __init__(self, model_path: str = "./finbert_financial_sentiment"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TRANSFORMERS else None
        self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        self.training_accuracy = "Unknown"

        print(f"🚀 Initializing Financial Fine-tuned FinBERT model...")
        print(f"📍 Model path: {model_path}")
        if self.device:
            print(f"🔧 Using device: {self.device}")

        self._load_model()

    def _load_model(self):
        """Load the fine-tuned financial sentiment model"""
        if not HAS_TRANSFORMERS:
            print("❌ transformers library not available - using fallback sentiment analysis")
            return

        if not os.path.exists(self.model_path):
            print(f"❌ Model path not found: {self.model_path}")
            print("Please ensure your fine-tuned financial model is in the correct location.")
            print("Run the fine-tuning script first to create the model.")
            return

        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            print("📥 Loading fine-tuned financial model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=3  # negative, neutral, positive
            )

            self.model.to(self.device)
            self.model.eval()

            # Load training info if available
            info_path = os.path.join(self.model_path, 'training_info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        training_info = json.load(f)
                        self.training_accuracy = training_info.get('test_accuracy', 'Unknown')
                        print(f"📊 Model training accuracy: {self.training_accuracy}")
                except Exception as e:
                    logger.warning(f"Could not load training info: {e}")

            print("✅ Financial Fine-tuned FinBERT model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            print("Make sure your model files are correctly saved and accessible.")
            self.model = None
            self.tokenizer = None

    def predict_sentiment_with_text(self, text: str) -> Dict:
        """Predict sentiment using fine-tuned financial model and return original text"""
        # Get the sentiment prediction
        sentiment_result = self.predict_sentiment(text)

        # Add the original text and a truncated version for display
        sentiment_result["original_text"] = text
        sentiment_result["display_text"] = self._truncate_text(text, 150)

        return sentiment_result

    def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment using fine-tuned financial model"""
        if not self.model or not self.tokenizer:
            return self._basic_sentiment_fallback(text)

        if not text or len(text.strip()) < 5:
            return {"sentiment": 0.0, "confidence": 0.0, "label": "neutral", "method": "empty_text"}

        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)

            # Tokenize input
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get results
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()

            # Get all class probabilities
            probs = predictions.cpu().numpy()[0]

            # Convert to sentiment score (-1 to +1 scale)
            if predicted_class == 2:  # positive
                sentiment_score = probs[2] - probs[0]  # positive_prob - negative_prob
            elif predicted_class == 0:  # negative
                sentiment_score = -(probs[0] - probs[2])  # -(negative_prob - positive_prob)
            else:  # neutral
                sentiment_score = (probs[2] - probs[0]) * 0.5  # Muted sentiment for neutral

            label = self.label_mapping[predicted_class]

            return {
                "sentiment": round(sentiment_score, 4),
                "confidence": round(confidence, 4),
                "label": label.upper(),
                "method": "fine_tuned_financial_finbert",
                "predicted_class": predicted_class,
                "class_probabilities": {
                    "negative": round(probs[0], 4),
                    "neutral": round(probs[1], 4),
                    "positive": round(probs[2], 4)
                }
            }

        except Exception as e:
            print(f"❌ Fine-tuned model prediction error: {e}")
            return self._basic_sentiment_fallback(text)

    def _truncate_text(self, text: str, max_length: int = 150) -> str:
        """Truncate text for display purposes"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for financial analysis"""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Preserve financial symbols and numbers
        # Keep $ symbols, percentages, and stock tickers
        text = re.sub(r'[^\w\s.,!?$%-]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _basic_sentiment_fallback(self, text: str) -> Dict:
        """Enhanced fallback sentiment analysis for financial text"""
        if not text:
            return {"sentiment": 0.0, "confidence": 0.0, "label": "NEUTRAL", "method": "fallback_empty"}

        text_lower = text.lower()

        # Enhanced financial sentiment keywords
        positive_words = [
            'buy', 'bull', 'bullish', 'up', 'rise', 'rising', 'gain', 'gains',
            'profit', 'profits', 'good', 'great', 'excellent', 'strong', 'growth',
            'increase', 'positive', 'optimistic', 'confidence', 'outperform',
            'beat', 'exceed', 'upgrade', 'recommend', 'target', 'momentum',
            'rally', 'surge', 'boom', 'expansion', 'breakthrough', 'success',
            'solid', 'robust', 'healthy', 'promising', 'favorable', 'dividend',
            'earnings beat', 'revenue growth', 'market share', 'innovation'
        ]

        negative_words = [
            'sell', 'bear', 'bearish', 'down', 'fall', 'falling', 'loss', 'losses',
            'bad', 'terrible', 'weak', 'decline', 'decrease', 'negative',
            'pessimistic', 'concern', 'underperform', 'miss', 'downgrade',
            'avoid', 'risk', 'crash', 'drop', 'plunge', 'collapse',
            'recession', 'crisis', 'warning', 'caution', 'debt', 'bankruptcy',
            'lawsuit', 'fraud', 'scandal', 'volatility', 'uncertainty'
        ]

        neutral_words = [
            'hold', 'neutral', 'stable', 'maintain', 'unchanged', 'steady',
            'sideways', 'range', 'consolidation', 'wait', 'monitor'
        ]

        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return {"sentiment": 0.0, "confidence": 0.0, "label": "NEUTRAL", "method": "fallback_empty"}

        # Calculate sentiment score
        net_sentiment = positive_count - negative_count
        sentiment_raw = net_sentiment / max(total_words, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_raw * 3))  # Scale and clamp

        # Adjust for neutral words
        if neutral_count > 0:
            sentiment_score *= (1 - (neutral_count / max(total_words, 1)) * 0.5)

        # Determine label and confidence
        if sentiment_score > 0.15:
            label = "POSITIVE"
            confidence = min(0.8, abs(sentiment_score) + 0.4)
        elif sentiment_score < -0.15:
            label = "NEGATIVE"
            confidence = min(0.8, abs(sentiment_score) + 0.4)
        else:
            label = "NEUTRAL"
            confidence = 0.6

        return {
            "sentiment": round(sentiment_score, 4),
            "confidence": round(confidence, 4),
            "label": label,
            "method": "enhanced_financial_fallback"
        }

class FinancialDataCollector:
    """Enhanced data collector for financial sentiment analysis with headline tracking"""

    def __init__(self, model_path: str = "./finbert_financial_sentiment"):
        self.sentiment_analyzer = FinancialFinBERTAnalyzer(model_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Initialize Reddit client
        self.reddit_client = None
        if HAS_REDDIT and API_KEYS["REDDIT_CLIENT_ID"] and API_KEYS["REDDIT_CLIENT_SECRET"]:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=API_KEYS["REDDIT_CLIENT_ID"],
                    client_secret=API_KEYS["REDDIT_CLIENT_SECRET"],
                    user_agent=API_KEYS["REDDIT_USER_AGENT"]
                )
                print("✅ Reddit client initialized")
            except Exception as e:
                print(f"❌ Error initializing Reddit client: {e}")
                self.reddit_client = None

    def _normalize_symbol(self, symbol: str) -> Dict[str, str]:
        """Extract and normalize symbol information"""
        symbol = symbol.upper().strip()

        # Handle different exchange formats
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            base_symbol = symbol.split('.')[0]
            exchange = 'NSE' if symbol.endswith('.NS') else 'BSE'
        elif symbol.endswith('.L'):
            base_symbol = symbol.split('.')[0]
            exchange = 'LSE'
        else:
            base_symbol = symbol
            exchange = 'US'

        # Get company information
        company_name = ""
        company_sector = ""

        if HAS_YFINANCE:
            try:
                ticker_obj = yf.Ticker(symbol)
                info = ticker_obj.info

                company_name = info.get('longName', '')
                if company_name:
                    company_name = re.sub(r'\b(Ltd|Limited|Inc|Corp|Corporation|Company|Co)\b\.?', '', company_name, flags=re.IGNORECASE)
                    company_name = company_name.strip()

                company_sector = info.get('sector', '')

                print(f"   📋 Company: {company_name}")
                print(f"   🏢 Sector: {company_sector}")

            except Exception as e:
                print(f"   ⚠️ Could not retrieve company info: {e}")

        return {
            'original_symbol': symbol,
            'base_symbol': base_symbol,
            'exchange': exchange,
            'company_name': company_name,
            'company_sector': company_sector
        }

    def get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get Reddit sentiment analysis with headlines"""
        print(f"🔍 Fetching Reddit data for {symbol}...")

        if not HAS_REDDIT or not self.reddit_client:
            print("❌ Reddit API not available - using demo data")
            symbol_info = self._normalize_symbol(symbol)
            demo_data = [
                {
                    "text": f"{symbol} quarterly results exceeded expectations, strong fundamentals",
                    "source": "r/investing",
                    "type": "post_title"
                },
                {
                    "text": f"Bullish on {symbol}, technical analysis shows upward momentum",
                    "source": "r/stocks",
                    "type": "comment"
                },
                {
                    "text": f"{symbol} dividend announcement positively received by market",
                    "source": "r/investing",
                    "type": "post_content"
                },
                {
                    "text": f"Concerned about {symbol} debt levels, might affect future growth",
                    "source": "r/SecurityAnalysis",
                    "type": "comment"
                },
                {
                    "text": f"{symbol} management guidance looks promising for next quarter",
                    "source": "r/investing",
                    "type": "post_title"
                }
            ]
            return self._analyze_texts_with_sources(demo_data, "Reddit (Demo)")

        symbol_info = self._normalize_symbol(symbol)
        text_data = []

        # Define subreddits based on exchange
        if symbol_info['exchange'] in ['NSE', 'BSE']:
            subreddits = ['IndiaInvestments', 'investing', 'stocks', 'SecurityAnalysis', 'IndianStreetBets']
        else:
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']

        # Search strategies
        search_queries = [symbol_info['base_symbol']]
        if symbol_info['company_name']:
            search_queries.append(f'"{symbol_info["company_name"]}"')

        for query in search_queries[:2]:
            for subreddit_name in subreddits:
                if len(text_data) >= 50:
                    break

                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)

                    for post in subreddit.search(query, limit=8, time_filter='month', sort='relevance'):
                        if self._is_relevant_post(post, symbol_info):
                            if post.title and len(post.title) > 20:
                                text_data.append({
                                    "text": post.title,
                                    "source": f"r/{subreddit_name}",
                                    "type": "post_title",
                                    "url": f"https://reddit.com{post.permalink}"
                                })

                            if hasattr(post, 'selftext') and post.selftext and len(post.selftext) > 30:
                                text_data.append({
                                    "text": post.selftext[:600],
                                    "source": f"r/{subreddit_name}",
                                    "type": "post_content",
                                    "url": f"https://reddit.com{post.permalink}"
                                })

                            # Get comments
                            try:
                                post.comments.replace_more(limit=0)
                                for comment in post.comments[:3]:
                                    if hasattr(comment, 'body') and len(comment.body) > 30:
                                        if self._is_relevant_comment(comment.body, symbol_info):
                                            text_data.append({
                                                "text": comment.body[:500],
                                                "source": f"r/{subreddit_name}",
                                                "type": "comment",
                                                "url": f"https://reddit.com{post.permalink}"
                                            })
                            except:
                                continue

                except Exception as e:
                    continue

        return self._analyze_texts_with_sources(text_data, "Reddit")

    def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment analysis with headlines"""
        print(f"🔍 Fetching News data for {symbol}...")
        text_data = []
        symbol_info = self._normalize_symbol(symbol)

        # Try NewsAPI
        if API_KEYS["NEWS_API_KEY"]:
            try:
                url = "https://newsapi.org/v2/everything"

                search_terms = [symbol_info['base_symbol']]
                if symbol_info['company_name']:
                    search_terms.append(f'"{symbol_info["company_name"]}"')

                query = ' OR '.join(search_terms)
                if symbol_info['exchange'] in ['NSE', 'BSE']:
                    query += ' AND (India OR NSE OR BSE OR stock)'
                else:
                    query += ' AND (stock OR financial OR earnings)'

                params = {
                    'q': query,
                    'apiKey': API_KEYS["NEWS_API_KEY"],
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 30,
                    'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                }

                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = article.get('content', '')
                        source_name = article.get('source', {}).get('name', 'Unknown')
                        url = article.get('url', '')

                        if title and len(title) > 20:
                            text_data.append({
                                "text": title,
                                "source": source_name,
                                "type": "headline",
                                "url": url,
                                "published_at": article.get('publishedAt', '')
                            })

                        if description and len(description) > 30:
                            text_data.append({
                                "text": description,
                                "source": source_name,
                                "type": "description",
                                "url": url,
                                "published_at": article.get('publishedAt', '')
                            })

                        if content and len(content) > 50:
                            text_data.append({
                                "text": content[:800],
                                "source": source_name,
                                "type": "content",
                                "url": url,
                                "published_at": article.get('publishedAt', '')
                            })

            except Exception as e:
                print(f"NewsAPI error: {e}")

        # Fallback to demo data
        if not text_data:
            company_name = symbol_info['company_name'] or symbol_info['base_symbol']
            text_data = [
                {
                    "text": f"{company_name} reports strong quarterly earnings growth",
                    "source": "Financial Times",
                    "type": "headline"
                },
                {
                    "text": f"{symbol_info['base_symbol']} stock rallies on positive analyst coverage",
                    "source": "Reuters",
                    "type": "headline"
                },
                {
                    "text": f"Institutional investors increase holdings in {company_name}",
                    "source": "Bloomberg",
                    "type": "headline"
                },
                {
                    "text": f"Market volatility creates uncertainty for {symbol_info['base_symbol']} investors",
                    "source": "CNBC",
                    "type": "headline"
                },
                {
                    "text": f"{company_name} management provides optimistic forward guidance",
                    "source": "Wall Street Journal",
                    "type": "headline"
                }
            ]

        return self._analyze_texts_with_sources(text_data, "News")

    def get_youtube_sentiment(self, symbol: str) -> Dict:
        """Get YouTube sentiment analysis with video titles"""
        print(f"🔍 Fetching YouTube data for {symbol}...")

        symbol_info = self._normalize_symbol(symbol)

        if not API_KEYS["YOUTUBE_API_KEY"]:
            company_name = symbol_info['company_name'] or symbol_info['base_symbol']
            demo_data = [
                {
                    "text": f"{symbol_info['base_symbol']} Technical Analysis - Bullish breakout pattern emerging",
                    "source": "Financial Analysis Channel",
                    "type": "video_title"
                },
                {
                    "text": f"Why {company_name} is my top pick for 2024 - Strong fundamentals",
                    "source": "Stock Guru",
                    "type": "video_title"
                },
                {
                    "text": f"{symbol_info['base_symbol']} Stock Analysis: Buy or Sell? Complete review",
                    "source": "Investment Insights",
                    "type": "video_title"
                },
                {
                    "text": f"Great analysis! I'm adding {symbol_info['base_symbol']} to my portfolio",
                    "source": "YouTube Comments",
                    "type": "comment"
                }
            ]
            return self._analyze_texts_with_sources(demo_data, "YouTube (Demo)")

        text_data = []
        try:
            search_queries = [
                f'"{symbol_info["base_symbol"]}" stock analysis',
                f'{symbol_info["company_name"]} investment' if symbol_info['company_name'] else None
            ]

            search_queries = [q for q in search_queries if q]

            for query in search_queries[:2]:
                search_url = "https://www.googleapis.com/youtube/v3/search"
                search_params = {
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': 15,
                    'key': API_KEYS["YOUTUBE_API_KEY"],
                    'order': 'relevance'
                }

                response = self.session.get(search_url, params=search_params)
                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('items', []):
                        snippet = item.get('snippet', {})
                        title = snippet.get('title', '')
                        description = snippet.get('description', '')
                        channel_title = snippet.get('channelTitle', 'Unknown Channel')
                        video_id = item.get('id', {}).get('videoId', '')
                        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

                        if title and len(title) > 20:
                            text_data.append({
                                "text": title,
                                "source": channel_title,
                                "type": "video_title",
                                "url": video_url
                            })
                        if description and len(description) > 30:
                            text_data.append({
                                "text": description[:400],
                                "source": channel_title,
                                "type": "video_description",
                                "url": video_url
                            })

        except Exception as e:
            print(f"YouTube API error: {e}")

        return self._analyze_texts_with_sources(text_data, "YouTube")

    def get_twitter_sentiment(self, symbol: str) -> Dict:
        """Get Twitter sentiment analysis with tweets"""
        print(f"🔍 Fetching Twitter data for {symbol}...")

        symbol_info = self._normalize_symbol(symbol)

        if not API_KEYS["TWITTER_BEARER_TOKEN"]:
            demo_data = [
                {
                    "text": f"${symbol_info['base_symbol']} breaking resistance, bullish momentum building 📈",
                    "source": "Twitter",
                    "type": "tweet"
                },
                {
                    "text": f"Excellent earnings report from ${symbol_info['base_symbol']}, going long",
                    "source": "Twitter",
                    "type": "tweet"
                },
                {
                    "text": f"${symbol_info['base_symbol']} showing weakness, considering profit taking",
                    "source": "Twitter",
                    "type": "tweet"
                },
                {
                    "text": f"${symbol_info['base_symbol']} fundamentals remain strong despite market volatility",
                    "source": "Twitter",
                    "type": "tweet"
                }
            ]
            return self._analyze_texts_with_sources(demo_data, "Twitter (Demo)")

        text_data = []
        try:
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {'Authorization': f'Bearer {API_KEYS["TWITTER_BEARER_TOKEN"]}'}

            search_terms = [f'${symbol_info["base_symbol"]}', symbol_info['base_symbol']]
            query = f'({" OR ".join(search_terms)}) (stock OR investment OR earnings OR financial)'

            if symbol_info['exchange'] in ['NSE', 'BSE']:
                query += ' (India OR NSE OR BSE)'

            query += ' -is:retweet lang:en'

            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'public_metrics,created_at'
            }

            response = self.session.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                for tweet in data.get('data', []):
                    text = tweet.get('text', '')
                    if text and len(text) > 25:
                        text_data.append({
                            "text": text,
                            "source": "Twitter",
                            "type": "tweet",
                            "created_at": tweet.get('created_at', '')
                        })

        except Exception as e:
            print(f"Twitter API error: {e}")

        return self._analyze_texts_with_sources(text_data, "Twitter")

    def get_rss_feeds_sentiment(self, symbol: str) -> Dict:
        """Get RSS feeds sentiment analysis with headlines"""
        print(f"🔍 Fetching RSS feeds data for {symbol}...")

        if not HAS_FEEDPARSER:
            print("⚠️ feedparser not available - using demo data")
            symbol_info = self._normalize_symbol(symbol)
            demo_data = [
                {
                    "text": f"{symbol} shows resilience amid market turbulence",
                    "source": "MarketWatch RSS",
                    "type": "rss_title"
                },
                {
                    "text": f"Analysts upgrade {symbol} following strong performance metrics",
                    "source": "Yahoo Finance RSS",
                    "type": "rss_title"
                }
            ]
            return self._analyze_texts_with_sources(demo_data, "RSS (Demo)")

        symbol_info = self._normalize_symbol(symbol)
        text_data = []

        # RSS feed URLs
        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.marketwatch.com/rss/topstories",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        ]

        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:10]:  # Limit to 10 entries per feed
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')

                    # Check if the entry is relevant to the symbol
                    if self._is_rss_entry_relevant(title + ' ' + summary, symbol_info):
                        if title and len(title) > 20:
                            text_data.append({
                                "text": title,
                                "source": feed.feed.get('title', 'RSS Feed'),
                                "type": "rss_title",
                                "url": entry.get('link', ''),
                                "published_at": entry.get('published', '')
                            })

                        if summary and len(summary) > 30:
                            text_data.append({
                                "text": summary[:500],
                                "source": feed.feed.get('title', 'RSS Feed'),
                                "type": "rss_summary",
                                "url": entry.get('link', ''),
                                "published_at": entry.get('published', '')
                            })

            except Exception as e:
                print(f"Error parsing RSS feed {feed_url}: {e}")
                continue

        return self._analyze_texts_with_sources(text_data, "RSS Feeds")

    def get_comprehensive_sentiment(self, symbol: str) -> Dict:
        """Get comprehensive sentiment analysis from all sources"""
        print(f"\n🎯 Starting comprehensive sentiment analysis for {symbol}")
        print("=" * 60)

        symbol_info = self._normalize_symbol(symbol)

        # Initialize results
        all_sources = {}
        overall_headlines = {
            "most_influential": [],
            "top_positive": [],
            "top_negative": [],
            "all_analyzed": []
        }

        # Get sentiment from all sources
        sources = [
            ("News", self.get_news_sentiment),
            ("Reddit", self.get_reddit_sentiment),
            ("Twitter", self.get_twitter_sentiment),
            ("YouTube", self.get_youtube_sentiment),
            ("RSS", self.get_rss_feeds_sentiment)
        ]

        total_count = 0
        weighted_sentiments = []
        source_weights = {"News": 0.3, "Reddit": 0.2, "Twitter": 0.2, "YouTube": 0.15, "RSS": 0.15}

        for source_name, source_func in sources:
            try:
                print(f"\n📊 Processing {source_name}...")
                result = source_func(symbol)
                all_sources[source_name] = result

                # Collect headlines from all sources
                if result.get('headlines'):
                    overall_headlines["most_influential"].extend(result['headlines'].get("most_influential", []))
                    overall_headlines["top_positive"].extend(result['headlines'].get("top_positive", []))
                    overall_headlines["top_negative"].extend(result['headlines'].get("top_negative", []))
                    overall_headlines["all_analyzed"].extend(result['headlines'].get("all_analyzed", []))

                # Calculate weighted sentiment
                if result.get('count', 0) > 0:
                    weight = source_weights.get(source_name, 0.1)
                    sentiment_score = result['sentiment']['score']
                    confidence = result['sentiment']['confidence']
                    weighted_sentiments.append({
                        'sentiment': sentiment_score,
                        'weight': weight,
                        'confidence': confidence,
                        'count': result['count']
                    })
                    total_count += result['count']

                print(f"   ✅ {source_name}: {result['count']} items, sentiment: {result['sentiment']['score']:.3f}")

            except Exception as e:
                print(f"   ❌ Error processing {source_name}: {e}")
                all_sources[source_name] = self._empty_result_with_headlines()

        # Calculate overall sentiment
        if weighted_sentiments:
            # Weighted average considering confidence and count
            total_weighted_score = 0
            total_weights = 0

            for item in weighted_sentiments:
                effective_weight = item['weight'] * item['confidence'] * min(item['count'] / 10, 1.0)
                total_weighted_score += item['sentiment'] * effective_weight
                total_weights += effective_weight

            overall_sentiment = total_weighted_score / total_weights if total_weights > 0 else 0
        else:
            overall_sentiment = 0

        # Determine overall label
        if overall_sentiment > 0.2:
            overall_label = "POSITIVE"
        elif overall_sentiment < -0.2:
            overall_label = "NEGATIVE"
        else:
            overall_label = "NEUTRAL"

        # Sort and limit headlines
        overall_headlines["most_influential"] = sorted(
            overall_headlines["most_influential"],
            key=lambda x: abs(x.get('sentiment_score', 0)),
            reverse=True
        )[:10]

        overall_headlines["top_positive"] = sorted(
            [h for h in overall_headlines["top_positive"] if h.get('sentiment_score', 0) > 0.15],
            key=lambda x: x.get('sentiment_score', 0),
            reverse=True
        )[:5]

        overall_headlines["top_negative"] = sorted(
            [h for h in overall_headlines["top_negative"] if h.get('sentiment_score', 0) < -0.15],
            key=lambda x: x.get('sentiment_score', 0)
        )[:5]

        # Generate trading signal
        trading_signal = self._generate_trading_signal(overall_sentiment, overall_headlines)

        return {
            "symbol": symbol_info['original_symbol'],
            "company_info": symbol_info,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_data_points": total_count,
            "overall_sentiment": {
                "score": round(overall_sentiment, 4),
                "label": overall_label,
                "confidence": self._calculate_overall_confidence(weighted_sentiments)
            },
            "source_breakdown": all_sources,
            "headlines": overall_headlines,
            "trading_signal": trading_signal,
            "model_info": {
                "method": "fine_tuned_financial_finbert",
                "training_accuracy": self.sentiment_analyzer.training_accuracy
            }
        }

    def display_analysis_results(self, results: Dict):
        """Display comprehensive analysis results with headlines"""
        print("\n" + "="*80)
        print(f"🎯 FINANCIAL SENTIMENT ANALYSIS REPORT")
        print("="*80)

        # Basic info
        print(f"📈 Symbol: {results['symbol']}")
        if results['company_info']['company_name']:
            print(f"🏢 Company: {results['company_info']['company_name']}")
        if results['company_info']['company_sector']:
            print(f"🏭 Sector: {results['company_info']['company_sector']}")
        print(f"🕐 Analysis Time: {results['analysis_timestamp']}")
        print(f"📊 Total Data Points: {results['total_data_points']}")

        # Overall sentiment
        sentiment = results['overall_sentiment']
        print(f"\n🎯 OVERALL SENTIMENT")
        print(f"   Score: {sentiment['score']:.4f}")
        print(f"   Label: {sentiment['label']}")
        print(f"   Confidence: {sentiment['confidence']:.4f}")

        # Trading signal
        signal = results['trading_signal']
        print(f"\n📈 TRADING SIGNAL")
        print(f"   Recommendation: {signal['recommendation']}")
        print(f"   Strength: {signal['strength']}")
        print(f"   Reason: {signal['reason']}")

        # Source breakdown
        print(f"\n📊 SOURCE BREAKDOWN")
        for source_name, source_data in results['source_breakdown'].items():
            sentiment_info = source_data['sentiment']
            print(f"   {source_name}: {source_data['count']} items | "
                  f"Sentiment: {sentiment_info['score']:.3f} ({sentiment_info['label']}) | "
                  f"Confidence: {sentiment_info['confidence']:.3f}")

        # Key headlines that influenced sentiment
        headlines = results['headlines']

        if headlines['most_influential']:
            print(f"\n🔥 MOST INFLUENTIAL HEADLINES/MESSAGES")
            for i, headline in enumerate(headlines['most_influential'][:5], 1):
                print(f"   {i}. [{headline['source']}] {headline['display_text']}")
                print(f"      Sentiment: {headline['sentiment_score']:.3f} | Confidence: {headline['confidence']:.3f}")
                if headline.get('url'):
                    print(f"      URL: {headline['url']}")
                print()

        if headlines['top_positive']:
            print(f"📈 TOP POSITIVE SIGNALS")
            for i, headline in enumerate(headlines['top_positive'][:3], 1):
                print(f"   {i}. [{headline['source']}] {headline['display_text']}")
                print(f"      Sentiment: +{headline['sentiment_score']:.3f}")
                print()

        if headlines['top_negative']:
            print(f"📉 TOP NEGATIVE SIGNALS")
            for i, headline in enumerate(headlines['top_negative'][:3], 1):
                print(f"   {i}. [{headline['source']}] {headline['display_text']}")
                print(f"      Sentiment: {headline['sentiment_score']:.3f}")
                print()

        # Model info
        print(f"🤖 MODEL INFORMATION")
        print(f"   Method: {results['model_info']['method']}")
        print(f"   Training Accuracy: {results['model_info']['training_accuracy']}")

        print("="*80)

    def _is_relevant_post(self, post, symbol_info: Dict) -> bool:
        """Check if a Reddit post is relevant to the stock"""
        title_lower = post.title.lower()
        selftext_lower = getattr(post, 'selftext', '').lower()
        full_text = f"{title_lower} {selftext_lower}"

        # Check for symbol mentions
        if symbol_info['base_symbol'].lower() in full_text:
            return True

        # Check for company name mentions
        if symbol_info['company_name']:
            company_words = symbol_info['company_name'].lower().split()
            if len(company_words) >= 2:
                matches = sum(1 for word in company_words if word in full_text and len(word) > 3)
                if matches >= 2:
                    return True

        return False

    def _is_relevant_comment(self, comment_text: str, symbol_info: Dict) -> bool:
        """Check if a comment is relevant to the stock"""
        comment_lower = comment_text.lower()

        if symbol_info['base_symbol'].lower() in comment_lower:
            return True

        if symbol_info['company_name']:
            company_words = symbol_info['company_name'].lower().split()
            for word in company_words:
                if len(word) > 3 and word in comment_lower:
                    return True

        return False

    def _is_rss_entry_relevant(self, entry_text: str, symbol_info: Dict) -> bool:
        """Check if an RSS entry is relevant to the stock"""
        entry_lower = entry_text.lower()

        # Check for symbol mentions
        if symbol_info['base_symbol'].lower() in entry_lower:
            return True

        # Check for company name mentions
        if symbol_info['company_name']:
            company_words = symbol_info['company_name'].lower().split()
            for word in company_words:
                if len(word) > 3 and word in entry_lower:
                    return True

        return False

    def _analyze_texts_with_sources(self, text_data: List[Dict], source: str) -> Dict:
        """Analyze texts with their sources and track headlines that influenced sentiment"""
        if not text_data:
            return self._empty_result_with_headlines()

        valid_text_data = []
        for item in text_data:
            if item.get('text') and isinstance(item['text'], str) and len(item['text'].strip()) > 15:
                valid_text_data.append(item)

        if not valid_text_data:
            return self._empty_result_with_headlines()

        # Analyze with fine-tuned model
        results = []
        sentiments = []
        confidences = []
        analyzed_items = []

        print(f"   📊 Analyzing {len(valid_text_data)} texts with fine-tuned Financial FinBERT...")

        for item in valid_text_data:
            try:
                text = item['text']
                result = self.sentiment_analyzer.predict_sentiment_with_text(text)
                results.append(result)
                sentiments.append(result['sentiment'])
                confidences.append(result['confidence'])

                # Store the analyzed item with metadata
                analyzed_item = {
                    "text": text,
                    "display_text": result.get('display_text', text[:150] + "..."),
                    "sentiment_score": result['sentiment'],
                    "confidence": result['confidence'],
                    "label": result['label'],
                    "source": item.get('source', 'Unknown'),
                    "type": item.get('type', 'text'),
                    "url": item.get('url', ''),
                    "published_at": item.get('published_at', ''),
                    "created_at": item.get('created_at', '')
                }
                analyzed_items.append(analyzed_item)

            except Exception as e:
                print(f"   ❌ Error analyzing text: {e}")
                continue

        if not sentiments:
            return self._empty_result_with_headlines()

        # Calculate metrics with fallback if numpy is not available
        if HAS_NUMPY:
            avg_sentiment = float(np.mean(sentiments))
            avg_confidence = float(np.mean(confidences))
            sentiment_std = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
        else:
            avg_sentiment = sum(sentiments) / len(sentiments)
            avg_confidence = sum(confidences) / len(confidences)
            # Simple std fallback
            mean_s = avg_sentiment
            variance = sum((s - mean_s) ** 2 for s in sentiments) / len(sentiments) if len(sentiments) > 1 else 0.0
            sentiment_std = variance ** 0.5

        # Determine overall label
        if avg_sentiment > 0.2:
            overall_label = "POSITIVE"
        elif avg_sentiment < -0.2:
            overall_label = "NEGATIVE"
        else:
            overall_label = "NEUTRAL"

        # Sort analyzed items by absolute sentiment score (most influential first)
        analyzed_items.sort(key=lambda x: abs(x['sentiment_score']), reverse=True)

        # Get top headlines/messages that influenced sentiment
        top_positive = [item for item in analyzed_items if item['sentiment_score'] > 0.15][:3]
        top_negative = [item for item in analyzed_items if item['sentiment_score'] < -0.15][:3]
        most_influential = analyzed_items[:5]

        return {
            "count": len(valid_text_data),
            "sentiment": {
                "score": round(avg_sentiment, 4),
                "confidence": round(avg_confidence, 4),
                "std_dev": round(sentiment_std, 4),
                "label": overall_label
            },
            "method": "fine_tuned_financial_finbert",
            "distribution": {
                "positive": sum(1 for s in sentiments if s > 0.15),
                "negative": sum(1 for s in sentiments if s < -0.15),
                "neutral": sum(1 for s in sentiments if -0.15 <= s <= 0.15)
            },
            "detailed_results": results[:5],
            "headlines": {
                "most_influential": most_influential,
                "top_positive": top_positive,
                "top_negative": top_negative,
                "all_analyzed": analyzed_items
            }
        }

    def _empty_result_with_headlines(self) -> Dict:
        """Return empty result structure with headlines"""
        return {
            "count": 0,
            "sentiment": {
                "score": 0,
                "confidence": 0,
                "std_dev": 0,
                "label": "NEUTRAL"
            },
            "method": "no_data",
            "distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "detailed_results": [],
            "headlines": {
                "most_influential": [],
                "top_positive": [],
                "top_negative": [],
                "all_analyzed": []
            }
        }

    def _generate_trading_signal(self, sentiment_score: float, headlines: Dict) -> Dict:
        """Generate trading signal based on sentiment and headlines per specified thresholds"""

        # Count strong signals across all analyzed items
        all_items = headlines.get('all_analyzed', [])
        strong_positive = len([h for h in all_items if h.get('sentiment_score', 0) > 0.4])
        strong_negative = len([h for h in all_items if h.get('sentiment_score', 0) < -0.4])

        # Apply thresholds
        if sentiment_score > 0.20 and strong_positive >= 2:
            recommendation = "STRONG BUY"
            strength = "High"
            reason = f"Score {sentiment_score:.3f} with {strong_positive} strong positive headlines (>0.4)"
        elif sentiment_score > 0.20:
            recommendation = "BUY"
            strength = "Medium"
            reason = f"Score {sentiment_score:.3f} exceeds +0.20"
        elif sentiment_score > 0.08:
            recommendation = "WEAK BUY"
            strength = "Low"
            reason = f"Score {sentiment_score:.3f} exceeds +0.08"
        elif -0.08 <= sentiment_score <= 0.08:
            recommendation = "HOLD"
            strength = "Low"
            reason = f"Score {sentiment_score:.3f} within [-0.08, +0.08]"
        elif sentiment_score < -0.20 and strong_negative >= 2:
            recommendation = "STRONG SELL"
            strength = "High"
            reason = f"Score {sentiment_score:.3f} with {strong_negative} strong negative headlines (<-0.4)"
        elif sentiment_score < -0.20:
            recommendation = "SELL"
            strength = "Medium"
            reason = f"Score {sentiment_score:.3f} below -0.20"
        elif sentiment_score < -0.08:
            recommendation = "WEAK SELL"
            strength = "Low"
            reason = f"Score {sentiment_score:.3f} below -0.08"
        else:
            recommendation = "HOLD"
            strength = "Low"
            reason = f"Score {sentiment_score:.3f} near neutral range"

        return {
            "recommendation": recommendation,
            "strength": strength,
            "reason": reason,
            "sentiment_score": sentiment_score,
            "strong_positive_signals": strong_positive,
            "strong_negative_signals": strong_negative
        }

    def _calculate_overall_confidence(self, weighted_sentiments: List[Dict]) -> float:
        """Calculate overall confidence based on all sources"""
        if not weighted_sentiments:
            return 0.0

        total_confidence = 0
        total_weight = 0

        for item in weighted_sentiments:
            weight = item['weight'] * min(item['count'] / 10, 1.0)  # Normalize by count
            total_confidence += item['confidence'] * weight
            total_weight += weight

        return round(total_confidence / total_weight if total_weight > 0 else 0.0, 4)

# Usage Example and Main Function
def main():
    """Main function to demonstrate the enhanced sentiment analysis"""

    # Initialize the collector
    collector = FinancialDataCollector()

    # Example usage
    symbol = input("Enter stock symbol (e.g., AAPL, TSLA, RELIANCE.NS): ").strip()

    if not symbol:
        symbol = "AAPL"  # Default symbol

    print(f"\n🚀 Starting comprehensive financial sentiment analysis for {symbol}")

    # Get comprehensive sentiment analysis
    results = collector.get_comprehensive_sentiment(symbol)

    # Display results
    collector.display_analysis_results(results)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_analysis_{symbol}_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()