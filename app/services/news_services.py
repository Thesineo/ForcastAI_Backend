"""
News Service for AI Financial Advisor
Enhanced news aggregation, sentiment analysis, and social media monitoring
Built on Alpha Vantage integration with multiple fallback sources
"""

import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time
import re
import json
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor
import nltk
from textblob import TextBlob

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_NLTK = True
except:
    HAS_NLTK = False

# Import configuration and cache
try:
    from app.core.config import settings
except ImportError:
    # Fallback settings for standalone testing
    class Settings:
        ALPHA_VANTAGE_API_KEY = None
        NEWS_API_KEY = None
        NEWSAPI_ORG_KEY = None
    settings = Settings()

from .cache import CacheService, CacheType, get_cache_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_POSITIVE = "Very Positive"
    POSITIVE = "Positive" 
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"

class NewsSource(Enum):
    """Available news sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    NEWSAPI_ORG = "newsapi_org"
    REDDIT = "reddit"
    TWITTER = "twitter"
    RSS_FEEDS = "rss_feeds"
    FALLBACK = "fallback"

class NewsCategory(Enum):
    """News categories"""
    EARNINGS = "earnings"
    ANALYSIS = "analysis"
    BREAKING = "breaking"
    INDUSTRY = "industry"
    REGULATORY = "regulatory"
    MERGER = "merger"
    DIVIDEND = "dividend"
    GENERAL = "general"

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    url: str
    description: str
    source: str
    published_at: datetime
    sentiment_label: SentimentLabel
    sentiment_score: float
    relevance_score: float
    ticker: str
    authors: List[str]
    category: NewsCategory
    keywords: List[str]
    impact_score: float
    social_mentions: int = 0
    engagement_score: float = 0.0
    credibility_score: float = 0.5
    market_hours: bool = False
    breaking_news: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result"""
    ticker: str
    overall_sentiment: SentimentLabel
    sentiment_score: float
    confidence: float
    total_articles: int
    positive_articles: int
    negative_articles: int
    neutral_articles: int
    trending_keywords: List[str]
    sentiment_trend: str  # "improving", "declining", "stable"
    impact_assessment: str  # "high", "medium", "low"
    news_volume: str  # "high", "normal", "low"
    credibility_weighted_score: float
    recent_breaking_news: List[NewsArticle]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class MarketNews:
    """Market-wide news summary"""
    market_sentiment: SentimentLabel
    sentiment_score: float
    top_headlines: List[NewsArticle]
    sector_sentiment: Dict[str, float]
    breaking_news: List[NewsArticle]
    trending_topics: List[str]
    market_movers: List[Dict]
    news_volume: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class NewsService:
    """Enhanced news aggregation and sentiment analysis service"""
    
    # Financial keywords for relevance scoring (enhanced from your original)
    FINANCIAL_KEYWORDS = {
        'high_impact': ['earnings', 'revenue', 'profit', 'loss', 'bankruptcy', 'merger', 'acquisition', 'ipo', 'split', 'dividend'],
        'medium_impact': ['forecast', 'guidance', 'outlook', 'analyst', 'rating', 'upgrade', 'downgrade', 'target'],
        'market_terms': ['stock', 'shares', 'trading', 'volume', 'market', 'investor', 'investment', 'sec', 'filing'],
        'sentiment_words': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'confident', 'concerned', 'positive', 'negative']
    }
    
    # News sources with API endpoints
    NEWS_SOURCES = {
        NewsSource.ALPHA_VANTAGE: "https://www.alphavantage.co/query",
        NewsSource.NEWSAPI_ORG: "https://newsapi.org/v2/everything",
    }
    
    def __init__(self, cache_service: CacheService = None):
        self.cache = cache_service or get_cache_service()
        self.session = self._create_session()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if HAS_NLTK else None
        self._initialize_keywords()
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'AI-Financial-Advisor/1.0 (News Service)',
            'Accept': 'application/json',
        })
        return session
    
    def _initialize_keywords(self):
        """Initialize keyword sets for analysis"""
        self.all_financial_keywords = set()
        for category in self.FINANCIAL_KEYWORDS.values():
            self.all_financial_keywords.update(category)
    
    async def fetch_news(self, 
                        ticker: str, 
                        limit: int = 10, 
                        sources: List[NewsSource] = None,
                        hours_back: int = 24) -> List[NewsArticle]:
        """Enhanced news fetching (improved version of your fetch_news function)"""
        
        if sources is None:
            sources = [NewsSource.ALPHA_VANTAGE, NewsSource.NEWSAPI_ORG]
        
        # Check cache first
        cache_key = f"{ticker}_{limit}_{hours_back}"
        cached_news = self.cache.get(CacheType.NEWS, cache_key)
        if cached_news:
            logger.info(f"Retrieved cached news for {ticker}")
            return [NewsArticle(**article) for article in cached_news]
        
        start_time = time.time()
        all_articles = []
        
        # Collect news from multiple sources concurrently
        tasks = []
        for source in sources:
            if source == NewsSource.ALPHA_VANTAGE and settings.ALPHA_VANTAGE_API_KEY:
                tasks.append(self._fetch_from_alpha_vantage_async(ticker, limit * 2, hours_back))
            elif source == NewsSource.NEWSAPI_ORG and getattr(settings, 'NEWSAPI_ORG_KEY', None):
                tasks.append(self._fetch_from_newsapi_async(ticker, limit * 2, hours_back))
        
        # Add fallback if no API sources available
        if not tasks:
            tasks.append(self._get_fallback_news_async(ticker, limit))
        
        # Execute tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif not isinstance(result, Exception):
                    logger.warning(f"Unexpected result type: {type(result)}")
        
        except Exception as e:
            logger.error(f"Error in concurrent news fetching: {e}")
            all_articles = await self._get_fallback_news_async(ticker, limit)
        
        # Remove duplicates and rank articles
        unique_articles = self._deduplicate_articles(all_articles)
        ranked_articles = self._rank_articles(unique_articles, ticker)[:limit]
        
        # Enhance articles with additional analysis
        enhanced_articles = []
        for article in ranked_articles:
            enhanced = self._enhance_article(article)
            enhanced_articles.append(enhanced)
        
        # Cache results
        cache_data = [asdict(article) for article in enhanced_articles]
        self.cache.set(CacheType.NEWS, cache_key, cache_data, ttl=900)  # 15 minutes
        
        execution_time = time.time() - start_time
        logger.info(f"Fetched {len(enhanced_articles)} articles for {ticker} in {execution_time:.2f}s")
        
        return enhanced_articles
    
    async def _fetch_from_alpha_vantage_async(self, 
                                            ticker: str, 
                                            limit: int, 
                                            hours_back: int) -> List[NewsArticle]:
        """Enhanced Alpha Vantage fetching (async version of your implementation)"""
        
        try:
            url = self.NEWS_SOURCES[NewsSource.ALPHA_VANTAGE]
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker.upper(),
                "apikey": settings.ALPHA_VANTAGE_API_KEY,
                "limit": min(200, limit * 2),
                "time_from": (datetime.now() - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M"),
                "sort": "LATEST"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    data = await response.json()
            
            # Error handling (your original logic)
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return []
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                await asyncio.sleep(1)
                return []
            
            articles = data.get("feed", [])
            if not articles:
                logger.warning("No articles from Alpha Vantage")
                return []
            
            # Process articles (enhanced version of your logic)
            processed_articles = []
            seen_urls = set()
            
            for article_data in articles:
                article = self._process_alpha_vantage_article(article_data, ticker, seen_urls)
                if article:
                    processed_articles.append(article)
            
            logger.info(f"Alpha Vantage returned {len(processed_articles)} articles for {ticker}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Alpha Vantage async fetch error: {e}")
            return []
    
    def _process_alpha_vantage_article(self, 
                                     article_data: Dict, 
                                     ticker: str, 
                                     seen_urls: set) -> Optional[NewsArticle]:
        """Process Alpha Vantage article (enhanced from your process_alpha_vantage_articles)"""
        
        try:
            url = article_data.get('url', '')
            if url in seen_urls or not url:
                return None
            seen_urls.add(url)
            
            title = article_data.get('title', '').strip()
            summary = article_data.get('summary', '').strip()
            
            if not title or not summary:
                return None
            
            # Enhanced sentiment analysis (your logic improved)
            sentiment_label, sentiment_score = self._analyze_article_sentiment(
                title, summary, article_data
            )
            
            # Enhanced relevance scoring (your calculate_relevance_score improved)
            relevance_score = self._calculate_enhanced_relevance_score(
                title, summary, ticker, article_data
            )
            
            # Skip low relevance articles
            if relevance_score < 3:
                return None
            
            # Determine category
            category = self._categorize_article(title, summary)
            
            # Extract keywords
            keywords = self._extract_keywords(title + " " + summary)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(title, summary, sentiment_score, relevance_score)
            
            # Parse published date
            published_at = self._parse_publish_date(article_data.get('time_published', ''))
            
            # Check if breaking news
            breaking_news = self._is_breaking_news(title, published_at)
            
            # Check market hours
            market_hours = self._is_market_hours(published_at)
            
            return NewsArticle(
                title=title,
                url=url,
                description=summary[:400] + "..." if len(summary) > 400 else summary,
                source=article_data.get('source', 'Alpha Vantage'),
                published_at=published_at,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                ticker=ticker.upper(),
                authors=article_data.get('authors', []),
                category=category,
                keywords=keywords,
                impact_score=impact_score,
                breaking_news=breaking_news,
                market_hours=market_hours,
                credibility_score=0.8  # Alpha Vantage is generally credible
            )
            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage article: {e}")
            return None
    
    async def _fetch_from_newsapi_async(self, 
                                      ticker: str, 
                                      limit: int, 
                                      hours_back: int) -> List[NewsArticle]:
        """Fetch from NewsAPI.org as alternative source"""
        
        try:
            if not getattr(settings, 'NEWSAPI_ORG_KEY', None):
                return []
            
            url = self.NEWS_SOURCES[NewsSource.NEWSAPI_ORG]
            params = {
                "q": f"{ticker} OR \"{ticker}\"",
                "apiKey": settings.NEWSAPI_ORG_KEY,
                "pageSize": min(100, limit),
                "sortBy": "publishedAt",
                "language": "en",
                "from": (datetime.now() - timedelta(hours=hours_back)).isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    data = await response.json()
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get("articles", [])
            processed_articles = []
            seen_urls = set()
            
            for article_data in articles:
                article = self._process_newsapi_article(article_data, ticker, seen_urls)
                if article:
                    processed_articles.append(article)
            
            logger.info(f"NewsAPI returned {len(processed_articles)} articles for {ticker}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    def _process_newsapi_article(self, 
                               article_data: Dict, 
                               ticker: str, 
                               seen_urls: set) -> Optional[NewsArticle]:
        """Process NewsAPI article"""
        
        try:
            url = article_data.get('url', '')
            if url in seen_urls or not url:
                return None
            seen_urls.add(url)
            
            title = article_data.get('title', '').strip()
            description = article_data.get('description', '').strip()
            
            if not title:
                return None
            
            # Sentiment analysis
            sentiment_label, sentiment_score = self._analyze_article_sentiment(title, description)
            
            # Relevance scoring
            relevance_score = self._calculate_enhanced_relevance_score(title, description, ticker)
            
            if relevance_score < 2:
                return None
            
            # Parse date
            published_at = datetime.fromisoformat(
                article_data.get('publishedAt', '').replace('Z', '+00:00')
            )
            
            return NewsArticle(
                title=title,
                url=url,
                description=description or title,
                source=article_data.get('source', {}).get('name', 'NewsAPI'),
                published_at=published_at,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                ticker=ticker.upper(),
                authors=[article_data.get('author', 'Unknown')] if article_data.get('author') else [],
                category=self._categorize_article(title, description),
                keywords=self._extract_keywords(title + " " + (description or "")),
                impact_score=self._calculate_impact_score(title, description, sentiment_score, relevance_score),
                breaking_news=self._is_breaking_news(title, published_at),
                market_hours=self._is_market_hours(published_at),
                credibility_score=0.7  # NewsAPI sources vary in credibility
            )
            
        except Exception as e:
            logger.error(f"Error processing NewsAPI article: {e}")
            return None
    
    async def _get_fallback_news_async(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Enhanced fallback news (async version of your get_fallback_news)"""
        
        current_time = datetime.now(timezone.utc)
        
        fallback_data = [
            {
                "title": f"{ticker.upper()} Technical Analysis: Key Support and Resistance Levels",
                "description": f"Comprehensive technical analysis of {ticker.upper()} stock showing key support at previous lows and resistance near recent highs. RSI indicators suggest current momentum.",
                "sentiment": SentimentLabel.NEUTRAL,
                "category": NewsCategory.ANALYSIS,
                "keywords": ["technical analysis", "support", "resistance", "RSI"]
            },
            {
                "title": f"{ticker.upper()} Earnings Preview: Analyst Expectations and Key Metrics",
                "description": f"Upcoming earnings report for {ticker.upper()} with analyst consensus estimates, revenue projections, and key metrics to watch.",
                "sentiment": SentimentLabel.POSITIVE,
                "category": NewsCategory.EARNINGS,
                "keywords": ["earnings", "analyst", "revenue", "forecast"]
            },
            {
                "title": f"{ticker.upper()} Market Position: Industry Analysis and Competitive Landscape",
                "description": f"Analysis of {ticker.upper()}'s market position within its sector, competitive advantages, and growth prospects in current market conditions.",
                "sentiment": SentimentLabel.NEUTRAL,
                "category": NewsCategory.INDUSTRY,
                "keywords": ["market position", "industry", "competitive", "growth"]
            },
            {
                "title": f"{ticker.upper()} Investment Outlook: Risk Assessment and Price Targets",
                "description": f"Investment analysis covering risk factors, potential catalysts, and price target updates for {ticker.upper()} from leading analysts.",
                "sentiment": SentimentLabel.POSITIVE,
                "category": NewsCategory.ANALYSIS,
                "keywords": ["investment", "risk", "price target", "catalyst"]
            },
            {
                "title": f"{ticker.upper()} Trading Volume Analysis: Institutional Activity and Trends",
                "description": f"Recent trading patterns for {ticker.upper()} showing institutional buying/selling activity, volume trends, and market maker positioning.",
                "sentiment": SentimentLabel.NEUTRAL,
                "category": NewsCategory.GENERAL,
                "keywords": ["volume", "institutional", "trading", "trends"]
            }
        ]
        
        articles = []
        for i, data in enumerate(fallback_data[:limit]):
            article = NewsArticle(
                title=data["title"],
                url=f"https://ai-advisor.com/analysis/{ticker.lower()}-{i+1}",
                description=data["description"],
                source="AI Financial Advisor",
                published_at=current_time - timedelta(hours=i),
                sentiment_label=data["sentiment"],
                sentiment_score=0.1 if data["sentiment"] == SentimentLabel.POSITIVE else -0.1 if data["sentiment"] == SentimentLabel.NEGATIVE else 0.0,
                relevance_score=10 - i,
                ticker=ticker.upper(),
                authors=["AI Research Team"],
                category=data["category"],
                keywords=data["keywords"],
                impact_score=0.5,
                credibility_score=0.6  # Fallback content
            )
            articles.append(article)
        
        return articles
    
    def _analyze_article_sentiment(self, 
                                 title: str, 
                                 description: str, 
                                 article_data: Dict = None) -> Tuple[SentimentLabel, float]:
        """Enhanced sentiment analysis"""
        
        try:
            text = f"{title} {description}".lower()
            
            # Use Alpha Vantage sentiment if available
            if article_data:
                av_sentiment = article_data.get('overall_sentiment_score', 0)
                if av_sentiment != 0:
                    return self._score_to_sentiment_label(av_sentiment), float(av_sentiment)
            
            # Use NLTK VADER if available
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(text)
                compound_score = scores['compound']
                return self._score_to_sentiment_label(compound_score), compound_score
            
            # Fallback to TextBlob
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                return self._score_to_sentiment_label(polarity), polarity
            except:
                pass
            
            # Simple keyword-based sentiment (last resort)
            positive_words = ['positive', 'good', 'great', 'excellent', 'strong', 'bullish', 'growth', 'profit', 'gain']
            negative_words = ['negative', 'bad', 'poor', 'weak', 'bearish', 'decline', 'loss', 'drop', 'fall']
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                return SentimentLabel.POSITIVE, 0.3
            elif neg_count > pos_count:
                return SentimentLabel.NEGATIVE, -0.3
            else:
                return SentimentLabel.NEUTRAL, 0.0
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentLabel.NEUTRAL, 0.0
    
    def _score_to_sentiment_label(self, score: float) -> SentimentLabel:
        """Convert sentiment score to label"""
        if score >= 0.5:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.1:
            return SentimentLabel.POSITIVE
        elif score <= -0.5:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_enhanced_relevance_score(self, 
                                          title: str, 
                                          description: str, 
                                          ticker: str,
                                          article_data: Dict = None) -> float:
        """Enhanced relevance scoring (improved from your calculate_relevance_score)"""
        
        try:
            content = f"{title} {description}".lower()
            ticker_lower = ticker.lower()
            score = 0.0
            
            # Alpha Vantage relevance score if available
            if article_data:
                ticker_sentiments = article_data.get('ticker_sentiment', [])
                for ts in ticker_sentiments:
                    if ts.get('ticker', '').upper() == ticker.upper():
                        score += float(ts.get('relevance_score', 0)) * 15
                        break
            
            # Ticker mentions (your original logic enhanced)
            ticker_mentions = content.count(ticker_lower)
            score += ticker_mentions * 8
            
            # Company name variations (if ticker is different from company name)
            # This could be enhanced with a company name lookup
            
            # Financial keyword scoring (your original enhanced)
            for category, keywords in self.FINANCIAL_KEYWORDS.items():
                category_weight = {'high_impact': 3, 'medium_impact': 2, 'market_terms': 1, 'sentiment_words': 1.5}.get(category, 1)
                for keyword in keywords:
                    if keyword in content:
                        score += category_weight
            
            # Title vs description weighting
            title_lower = title.lower()
            if ticker_lower in title_lower:
                score += 5  # Higher weight for ticker in title
            
            # Recency bonus
            if article_data and 'time_published' in article_data:
                try:
                    pub_time = datetime.strptime(article_data['time_published'], '%Y%m%dT%H%M%S')
                    hours_ago = (datetime.now() - pub_time).total_seconds() / 3600
                    if hours_ago < 24:
                        score += max(0, (24 - hours_ago) / 24 * 3)  # Up to 3 points for recent news
                except:
                    pass
            
            # Breaking news bonus
            if any(word in content for word in ['breaking', 'urgent', 'alert', 'just in']):
                score += 5
            
            return min(score, 50)  # Cap at 50
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def _categorize_article(self, title: str, description: str) -> NewsCategory:
        """Categorize article based on content"""
        
        content = f"{title} {description}".lower()
        
        if any(word in content for word in ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'revenue']):
            return NewsCategory.EARNINGS
        elif any(word in content for word in ['merger', 'acquisition', 'deal', 'takeover']):
            return NewsCategory.MERGER
        elif any(word in content for word in ['dividend', 'payout', 'yield']):
            return NewsCategory.DIVIDEND
        elif any(word in content for word in ['sec', 'regulatory', 'compliance', 'filing']):
            return NewsCategory.REGULATORY
        elif any(word in content for word in ['breaking', 'urgent', 'alert', 'just in']):
            return NewsCategory.BREAKING
        elif any(word in content for word in ['analysis', 'forecast', 'outlook', 'target', 'rating']):
            return NewsCategory.ANALYSIS
        elif any(word in content for word in ['industry', 'sector', 'market', 'competition']):
            return NewsCategory.INDUSTRY
        else:
            return NewsCategory.GENERAL
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract relevant keywords from text"""
        
        try:
            text = text.lower()
            
            # Find financial keywords
            found_keywords = []
            for category in self.FINANCIAL_KEYWORDS.values():
                for keyword in category:
                    if keyword in text and keyword not in found_keywords:
                        found_keywords.append(keyword)
            
            # Add company/ticker specific terms
            company_terms = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
            for term in company_terms[:3]:  # Limit company terms
                if term not in found_keywords:
                    found_keywords.append(term.lower())
            
            return found_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _calculate_impact_score(self, 
                              title: str, 
                              description: str, 
                              sentiment_score: float, 
                              relevance_score: float) -> float:
        """Calculate potential market impact score"""
        
        try:
            content = f"{title} {description}".lower()
            
            # Base impact from relevance and sentiment strength
            base_impact = (relevance_score / 50) * 0.4 + abs(sentiment_score) * 0.3
            
            # High impact keywords
            high_impact_terms = [
                'earnings', 'revenue', 'profit', 'loss', 'bankruptcy', 'merger', 'acquisition',
                'fda approval', 'lawsuit', 'investigation', 'sec', 'dividend', 'stock split'
            ]
            
            impact_bonus = 0
            for term in high_impact_terms:
                if term in content:
                    impact_bonus += 0.1
            
            # Breaking news multiplier
            if any(word in content for word in ['breaking', 'urgent', 'alert']):
                base_impact *= 1.5
            
            return min(1.0, base_impact + impact_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.3  # Default medium impact
    
    def _parse_publish_date(self, date_str: str) -> datetime:
        """Parse publish date from various formats"""
        
        try:
            # Alpha Vantage format: YYYYMMDDTHHMMSS
            if len(date_str) == 15 and 'T' in date_str:
                return datetime.strptime(date_str, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
            
            # ISO format
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Fallback to current time
            return datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            return datetime.now(timezone.utc)
    
    def _is_breaking_news(self, title: str, published_at: datetime) -> bool:
        """Determine if article is breaking news"""
        try:
            # Check for breaking news keywords
            breaking_keywords = ['breaking', 'urgent', 'alert', 'just in', 'live:', 'update:']
            title_lower = title.lower()
            
            has_breaking_keyword = any(keyword in title_lower for keyword in breaking_keywords)
            
            # Check if published in last 2 hours
            hours_ago = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
            is_recent = hours_ago <= 2
            
            return has_breaking_keyword or (is_recent and any(word in title_lower for word in ['announces', 'reports', 'files']))
            
        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")
            return False
    
    def _is_market_hours(self, published_at: datetime) -> bool:
        """Check if article was published during market hours"""
        try:
            # Convert to Eastern Time (US market timezone)
            eastern = published_at.astimezone(timezone(timedelta(hours=-5)))
            
            # Check if weekday (0=Monday, 6=Sunday)
            if eastern.weekday() >= 5:
                return False
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= eastern <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL and title similarity"""
        try:
            seen_urls = set()
            seen_titles = set()
            unique_articles = []
            
            for article in articles:
                # Skip if URL already seen
                if article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                
                # Check title similarity (simple approach)
                title_words = set(article.title.lower().split())
                is_duplicate = False
                
                for seen_title in seen_titles:
                    seen_words = set(seen_title.split())
                    overlap = len(title_words & seen_words)
                    total_words = len(title_words | seen_words)
                    
                    # If >70% word overlap, consider duplicate
                    if total_words > 0 and overlap / total_words > 0.7:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_titles.add(article.title.lower())
                    unique_articles.append(article)
            
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error deduplicating articles: {e}")
            return articles
    
    def _rank_articles(self, articles: List[NewsArticle], ticker: str) -> List[NewsArticle]:
        """Rank articles by relevance, impact, and recency"""
        try:
            def ranking_score(article: NewsArticle) -> float:
                # Weighted scoring
                relevance_weight = 0.3
                impact_weight = 0.25
                sentiment_weight = 0.15
                recency_weight = 0.2
                credibility_weight = 0.1
                
                # Recency score (newer = higher score)
                hours_ago = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
                recency_score = max(0, (48 - hours_ago) / 48)  # 0-1 score over 48 hours
                
                # Breaking news bonus
                breaking_bonus = 0.2 if article.breaking_news else 0
                
                total_score = (
                    article.relevance_score / 50 * relevance_weight +
                    article.impact_score * impact_weight +
                    abs(article.sentiment_score) * sentiment_weight +
                    recency_score * recency_weight +
                    article.credibility_score * credibility_weight +
                    breaking_bonus
                )
                
                return total_score
            
            ranked = sorted(articles, key=ranking_score, reverse=True)
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking articles: {e}")
            return articles
    
    def _enhance_article(self, article: NewsArticle) -> NewsArticle:
        """Enhance article with additional metadata"""
        try:
            # Calculate social engagement score (placeholder - would integrate with social APIs)
            article.engagement_score = min(1.0, article.relevance_score / 30 + article.impact_score)
            
            # Estimate social mentions (placeholder)
            article.social_mentions = int(article.engagement_score * 100 * article.credibility_score)
            
            return article
            
        except Exception as e:
            logger.error(f"Error enhancing article: {e}")
            return article
    
    async def analyze_sentiment(self, 
                              ticker: str, 
                              hours_back: int = 24, 
                              min_articles: int = 5) -> Optional[SentimentAnalysis]:
        """Comprehensive sentiment analysis for a ticker"""
        
        try:
            # Get recent news articles
            articles = await self.fetch_news(ticker, limit=50, hours_back=hours_back)
            
            if len(articles) < min_articles:
                logger.warning(f"Insufficient articles for sentiment analysis: {len(articles)}")
                return None
            
            # Analyze sentiment distribution
            sentiments = [article.sentiment_score for article in articles]
            positive_articles = [a for a in articles if a.sentiment_score > 0.1]
            negative_articles = [a for a in articles if a.sentiment_score < -0.1]
            neutral_articles = [a for a in articles if -0.1 <= a.sentiment_score <= 0.1]
            
            # Calculate weighted sentiment (weight by credibility and impact)
            weighted_sentiment = sum(
                a.sentiment_score * a.credibility_score * a.impact_score 
                for a in articles
            ) / sum(a.credibility_score * a.impact_score for a in articles)
            
            # Overall sentiment label
            overall_sentiment = self._score_to_sentiment_label(weighted_sentiment)
            
            # Calculate confidence based on article volume and consistency
            sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
            confidence = min(0.95, max(0.1, 1 - sentiment_std))
            
            # Extract trending keywords
            all_keywords = []
            for article in articles:
                all_keywords.extend(article.keywords)
            
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            trending_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            trending_keywords = [k[0] for k in trending_keywords]
            
            # Determine sentiment trend (compare recent vs older articles)
            recent_articles = [a for a in articles if (datetime.now(timezone.utc) - a.published_at).total_seconds() < 12*3600]
            older_articles = [a for a in articles if (datetime.now(timezone.utc) - a.published_at).total_seconds() >= 12*3600]
            
            sentiment_trend = "stable"
            if recent_articles and older_articles:
                recent_sentiment = np.mean([a.sentiment_score for a in recent_articles])
                older_sentiment = np.mean([a.sentiment_score for a in older_articles])
                
                if recent_sentiment > older_sentiment + 0.1:
                    sentiment_trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    sentiment_trend = "declining"
            
            # Impact assessment
            avg_impact = np.mean([a.impact_score for a in articles])
            impact_assessment = "high" if avg_impact > 0.7 else "medium" if avg_impact > 0.4 else "low"
            
            # News volume assessment
            news_volume = "high" if len(articles) > 20 else "normal" if len(articles) > 10 else "low"
            
            # Breaking news
            breaking_news = [a for a in articles if a.breaking_news][:5]
            
            analysis = SentimentAnalysis(
                ticker=ticker.upper(),
                overall_sentiment=overall_sentiment,
                sentiment_score=weighted_sentiment,
                confidence=confidence,
                total_articles=len(articles),
                positive_articles=len(positive_articles),
                negative_articles=len(negative_articles),
                neutral_articles=len(neutral_articles),
                trending_keywords=trending_keywords,
                sentiment_trend=sentiment_trend,
                impact_assessment=impact_assessment,
                news_volume=news_volume,
                credibility_weighted_score=weighted_sentiment,
                recent_breaking_news=breaking_news
            )
            
            # Cache sentiment analysis
            cache_key = f"{ticker}_sentiment_{hours_back}h"
            self.cache.set(CacheType.SENTIMENT, cache_key, asdict(analysis), ttl=1800)
            
            logger.info(f"Sentiment analysis for {ticker}: {overall_sentiment.value} ({weighted_sentiment:.3f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {ticker}: {e}")
            return None
    
    async def get_market_news(self, limit: int = 20) -> Optional[MarketNews]:
        """Get overall market news and sentiment"""
        
        try:
            # Major market tickers to analyze
            major_tickers = ['SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            all_articles = []
            sentiment_scores = []
            
            # Collect news from major tickers
            tasks = [self.fetch_news(ticker, limit=10, hours_back=24) for ticker in major_tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, list):
                    all_articles.extend(result)
                    # Collect sentiment scores for overall market sentiment
                    ticker_sentiment = [a.sentiment_score for a in result]
                    if ticker_sentiment:
                        sentiment_scores.extend(ticker_sentiment)
            
            # Remove duplicates and get top articles
            unique_articles = self._deduplicate_articles(all_articles)
            top_articles = sorted(unique_articles, key=lambda x: x.impact_score, reverse=True)[:limit]
            
            # Calculate overall market sentiment
            market_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
            market_sentiment = self._score_to_sentiment_label(market_sentiment_score)
            
            # Get breaking news
            breaking_news = [a for a in unique_articles if a.breaking_news][:10]
            
            # Extract trending topics
            all_keywords = []
            for article in top_articles:
                all_keywords.extend(article.keywords)
            
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            trending_topics = [k for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)][:15]
            
            # Mock sector sentiment (would be calculated from sector-specific news)
            sector_sentiment = {
                'Technology': np.mean([a.sentiment_score for a in top_articles if 'tech' in ' '.join(a.keywords).lower()]) or 0,
                'Healthcare': np.mean([a.sentiment_score for a in top_articles if 'health' in ' '.join(a.keywords).lower()]) or 0,
                'Financial': np.mean([a.sentiment_score for a in top_articles if any(w in ' '.join(a.keywords).lower() for w in ['bank', 'financial'])]) or 0,
                'Energy': np.mean([a.sentiment_score for a in top_articles if 'energy' in ' '.join(a.keywords).lower()]) or 0
            }
            
            # Market movers (mock data - would integrate with market data service)
            market_movers = [
                {'symbol': 'AAPL', 'change_percent': 2.5, 'news_sentiment': 0.3},
                {'symbol': 'TSLA', 'change_percent': -1.8, 'news_sentiment': -0.2},
                {'symbol': 'MSFT', 'change_percent': 1.2, 'news_sentiment': 0.1}
            ]
            
            market_news = MarketNews(
                market_sentiment=market_sentiment,
                sentiment_score=market_sentiment_score,
                top_headlines=top_articles[:10],
                sector_sentiment=sector_sentiment,
                breaking_news=breaking_news,
                trending_topics=trending_topics,
                market_movers=market_movers,
                news_volume=len(unique_articles)
            )
            
            # Cache market news
            self.cache.set(CacheType.NEWS, "market_summary", asdict(market_news), ttl=900)  # 15 minutes
            
            logger.info(f"Market news summary: {market_sentiment.value} sentiment, {len(top_articles)} articles")
            return market_news
            
        except Exception as e:
            logger.error(f"Error getting market news: {e}")
            return None
    
    async def get_news_summary(self, ticker: str) -> Dict[str, Any]:
        """Enhanced news summary (improved version of your get_news_summary)"""
        
        try:
            # Get sentiment analysis
            sentiment_analysis = await self.analyze_sentiment(ticker, hours_back=24)
            
            if not sentiment_analysis:
                return {"error": "No news data available"}
            
            # Get recent articles for headlines
            articles = await self.fetch_news(ticker, limit=5, hours_back=24)
            headlines = [a.title for a in articles[:3]] if articles else []
            
            summary = {
                "ticker": ticker.upper(),
                "sentiment_analysis": {
                    "overall_sentiment": sentiment_analysis.overall_sentiment.value,
                    "sentiment_score": round(sentiment_analysis.sentiment_score, 3),
                    "confidence": round(sentiment_analysis.confidence, 3),
                    "sentiment_trend": sentiment_analysis.sentiment_trend
                },
                "article_stats": {
                    "total_articles": sentiment_analysis.total_articles,
                    "positive_articles": sentiment_analysis.positive_articles,
                    "negative_articles": sentiment_analysis.negative_articles,
                    "neutral_articles": sentiment_analysis.neutral_articles
                },
                "key_insights": {
                    "trending_keywords": sentiment_analysis.trending_keywords[:5],
                    "impact_assessment": sentiment_analysis.impact_assessment,
                    "news_volume": sentiment_analysis.news_volume,
                    "breaking_news_count": len(sentiment_analysis.recent_breaking_news)
                },
                "latest_headlines": headlines,
                "data_quality": {
                    "credibility_score": round(sentiment_analysis.credibility_weighted_score, 3),
                    "data_source": "Alpha Vantage" if settings.ALPHA_VANTAGE_API_KEY else "Multiple Sources"
                },
                "timestamp": sentiment_analysis.timestamp.isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating news summary for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    async def check_api_status(self) -> Dict[str, Any]:
        """Enhanced API status check (improved version of your check_api_status)"""
        
        status_results = {}
        
        # Check Alpha Vantage
        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                url = self.NEWS_SOURCES[NewsSource.ALPHA_VANTAGE]
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": "AAPL",
                    "apikey": settings.ALPHA_VANTAGE_API_KEY,
                    "limit": 1
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        data = await response.json()
                
                if "Error Message" in data:
                    status_results["alpha_vantage"] = {
                        "status": "error",
                        "message": f"API Error: {data['Error Message']}"
                    }
                elif "Note" in data:
                    status_results["alpha_vantage"] = {
                        "status": "warning", 
                        "message": f"Rate Limited: {data['Note']}"
                    }
                elif "feed" in data:
                    status_results["alpha_vantage"] = {
                        "status": "success",
                        "message": "Alpha Vantage API working",
                        "articles_available": len(data["feed"])
                    }
                else:
                    status_results["alpha_vantage"] = {
                        "status": "warning",
                        "message": "API working but no articles returned"
                    }
                    
            except Exception as e:
                status_results["alpha_vantage"] = {
                    "status": "error",
                    "message": f"Connection failed: {str(e)}"
                }
        else:
            status_results["alpha_vantage"] = {
                "status": "disabled",
                "message": "No Alpha Vantage API key configured"
            }
        
        # Check NewsAPI if available
        if getattr(settings, 'NEWSAPI_ORG_KEY', None):
            try:
                url = self.NEWS_SOURCES[NewsSource.NEWSAPI_ORG]
                params = {
                    "q": "AAPL",
                    "apiKey": settings.NEWSAPI_ORG_KEY,
                    "pageSize": 1
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        data = await response.json()
                
                if data.get("status") == "ok":
                    status_results["newsapi"] = {
                        "status": "success",
                        "message": "NewsAPI working",
                        "articles_available": data.get("totalResults", 0)
                    }
                else:
                    status_results["newsapi"] = {
                        "status": "error",
                        "message": data.get("message", "Unknown error")
                    }
                    
            except Exception as e:
                status_results["newsapi"] = {
                    "status": "error",
                    "message": f"NewsAPI connection failed: {str(e)}"
                }
        else:
            status_results["newsapi"] = {
                "status": "disabled",
                "message": "No NewsAPI key configured"
            }
        
        # Overall status
        active_sources = [k for k, v in status_results.items() if v["status"] == "success"]
        overall_status = "healthy" if active_sources else "degraded" if any(v["status"] == "warning" for v in status_results.values()) else "unhealthy"
        
        return {
            "service": "news_service",
            "overall_status": overall_status,
            "active_sources": active_sources,
            "source_details": status_results,
            "fallback_available": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Synchronous health check for the news service"""
        try:
            start_time = time.time()
            
            # Test basic functionality with fallback news
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                fallback_news = loop.run_until_complete(self._get_fallback_news_async("AAPL", 3))
                fallback_working = len(fallback_news) > 0
            except Exception as e:
                fallback_working = False
                logger.error(f"Fallback news test failed: {e}")
            finally:
                loop.close()
            
            # Test sentiment analysis
            try:
                test_sentiment = self._analyze_article_sentiment("Positive earnings report", "Strong growth expected")
                sentiment_working = test_sentiment[0] == SentimentLabel.POSITIVE
            except Exception as e:
                sentiment_working = False
                logger.error(f"Sentiment analysis test failed: {e}")
            
            response_time = time.time() - start_time
            
            # Component status
            components = {
                "fallback_news": "healthy" if fallback_working else "unhealthy",
                "sentiment_analysis": "healthy" if sentiment_working else "unhealthy",
                "cache_service": "healthy" if self.cache.get_stats().get('status') == 'connected' else 'unhealthy',
                "nltk_sentiment": "available" if HAS_NLTK else "unavailable"
            }
            
            overall_healthy = fallback_working and sentiment_working
            
            return {
                "service": "news_service", 
                "status": "healthy" if overall_healthy else "unhealthy",
                "components": components,
                "api_keys": {
                    "alpha_vantage": bool(settings.ALPHA_VANTAGE_API_KEY),
                    "newsapi": bool(getattr(settings, 'NEWSAPI_ORG_KEY', None))
                },
                "response_time": round(response_time, 3),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "service": "news_service",
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)

# Legacy compatibility functions (enhanced versions of your original functions)
def analyze_sentiment(news_list: List[Dict]) -> Dict[str, Any]:
    """
    Legacy analyze_sentiment function for backward compatibility
    Enhanced version of your sentiment.py analyze_sentiment function
    """
    service = get_news_service()
    return service.analyze_sentiment_legacy(news_list)

async def analyze_news_sentiment(articles: List[NewsArticle]) -> Dict[str, Any]:
    """
    Enhanced bulk sentiment analysis for NewsArticle objects
    """
    service = get_news_service()
    return await service.analyze_news_sentiment_bulk(articles)

# Legacy compatibility functions (enhanced versions of your original functions)
async def fetch_news(ticker: str, limit: int = 5) -> List[Dict]:
    """Legacy fetch_news function for backward compatibility"""
    service = NewsService()
    articles = await service.fetch_news(ticker, limit)
    
    # Convert to dict format similar to your original
    return [
        {
            "title": a.title,
            "url": a.url,
            "publishedAt": a.published_at.isoformat(),
            "description": a.description,
            "source": a.source,
            "sentiment": a.sentiment_label.value,
            "sentiment_score": a.sentiment_score,
            "relevance_score": a.relevance_score,
            "ticker": a.ticker,
            "authors": ", ".join(a.authors) if a.authors else "Unknown",
            "category": a.category.value
        }
        for a in articles
    ]

async def get_news_summary(ticker: str) -> Dict[str, Any]:
    """Legacy get_news_summary function"""
    service = NewsService()
    return await service.get_news_summary(ticker)

async def test_alpha_vantage_api():
    """Enhanced test function (async version of your test_alpha_vantage_api)"""
    service = NewsService()
    
    print("🔍 Testing Enhanced News Service...")
    print(f"📊 Alpha Vantage API Key: {'✅ Available' if settings.ALPHA_VANTAGE_API_KEY else '❌ Missing'}")
    print(f"📊 NewsAPI Key: {'✅ Available' if getattr(settings, 'NEWSAPI_ORG_KEY', None) else '❌ Missing'}")
    print(f"🧠 NLTK Sentiment: {'✅ Available' if HAS_NLTK else '❌ Missing'}")
    
    # Test API status
    api_status = await service.check_api_status()
    print(f"\n📡 Overall API Status: {api_status['overall_status'].upper()}")
    
    # Test news fetching
    test_tickers = ["AAPL", "TSLA", "MSFT"]
    
    for ticker in test_tickers:
        print(f"\n📈 Testing {ticker}:")
        try:
            articles = await service.fetch_news(ticker, 3)
            print(f"  📰 Articles retrieved: {len(articles)}")
            
            for i, article in enumerate(articles[:2], 1):
                print(f"  {i}. {article.title[:60]}...")
                print(f"     💭 Sentiment: {article.sentiment_label.value} ({article.sentiment_score:.3f})")
                print(f"     📊 Relevance: {article.relevance_score:.1f} | Impact: {article.impact_score:.2f}")
                print(f"     🏷️  Category: {article.category.value}")
            
            # Test sentiment analysis
            sentiment = await service.analyze_sentiment(ticker, hours_back=24)
            if sentiment:
                print(f"  📊 Overall Sentiment: {sentiment.overall_sentiment.value} ({sentiment.sentiment_score:.3f})")
                print(f"  🎯 Confidence: {sentiment.confidence:.2f} | Trend: {sentiment.sentiment_trend}")
                print(f"  📈 Articles: {sentiment.positive_articles}+ {sentiment.negative_articles}- {sentiment.neutral_articles}=")
                print(f"  🔥 Trending: {', '.join(sentiment.trending_keywords[:3])}")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n✅ Enhanced News Service Test Completed!")

# Global service instance
_news_service = None

def get_news_service() -> NewsService:
    """Get global news service instance"""
    global _news_service
    if _news_service is None:
        _news_service = NewsService()
    return _news_service

# Convenience functions
async def get_stock_news(ticker: str, limit: int = 10) -> List[NewsArticle]:
    """Convenience function for getting stock news"""
    service = get_news_service()
    return await service.fetch_news(ticker, limit)

async def get_sentiment_analysis(ticker: str, hours_back: int = 24) -> Optional[SentimentAnalysis]:
    """Convenience function for sentiment analysis"""
    service = get_news_service()
    return await service.analyze_sentiment(ticker, hours_back)

async def get_market_overview() -> Optional[MarketNews]:
    """Convenience function for market news overview"""
    service = get_news_service()
    return await service.get_market_news()

if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    async def main():
        print("🚀 Testing Enhanced News Service...\n")
        
        # Health check first
        service = NewsService()
        health = service.health_check()
        print(f"🏥 Health Check: {health['status'].upper()}")
        print(f"📊 Components: {health['components']}")
        
        # API status check
        api_status = await service.check_api_status()
        print(f"📡 API Status: {api_status['overall_status'].upper()}")
        print(f"🔄 Active Sources: {api_status['active_sources']}")
        
        print("\n" + "="*60 + "\n")
        
        # Test comprehensive news analysis
        await test_alpha_vantage_api()
        
        # Clean up
        service.close()
    
    # Run the async test
    asyncio.run(main())