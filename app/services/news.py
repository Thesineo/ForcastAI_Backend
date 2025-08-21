import requests
from app.core.config import settings
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import time

# Set up logging
logger = logging.getLogger(__name__)

def fetch_news(ticker: str, limit: int = 5):
    """
    Fetch news for a given ticker using Alpha Vantage API
    """
    # Check if Alpha Vantage API key is available
    if not settings.ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY not found in settings")
        return get_fallback_news(ticker, limit)
    
    try:
        # Try Alpha Vantage News Sentiment
        news_data = fetch_from_alpha_vantage(ticker, settings.ALPHA_VANTAGE_API_KEY, limit)
        if news_data and len(news_data) > 0:
            logger.info(f"Successfully retrieved {len(news_data)} articles from Alpha Vantage for {ticker}")
            return news_data
            
    except Exception as e:
        logger.error(f"Alpha Vantage API failed: {str(e)}")
    
    # Final fallback: Return mock data
    logger.info(f"Using fallback news data for {ticker}")
    return get_fallback_news(ticker, limit)


def fetch_from_alpha_vantage(ticker: str, api_key: str, limit: int = 5):
    """
    Fetch news from Alpha Vantage News Sentiment API
    """
    try:
        # Alpha Vantage News Sentiment endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "apikey": api_key,
            "limit": min(200, limit * 4),  # Get more to filter from
            "time_from": (datetime.now() - timedelta(days=30)).strftime("%Y%m%dT%H%M"),
            "sort": "LATEST"
        }
        
        logger.info(f"Fetching news for {ticker} from Alpha Vantage...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors or rate limiting
        if "Error Message" in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            return None
            
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
            # Brief pause for rate limiting, then continue with fallback
            time.sleep(1)
            return None
            
        # Extract news articles
        articles = data.get("feed", [])
        logger.info(f"Retrieved {len(articles)} raw articles from Alpha Vantage")
        
        if not articles:
            logger.warning("No articles returned from Alpha Vantage")
            return None
            
        # Process and format articles
        processed_articles = process_alpha_vantage_articles(articles, ticker, limit)
        logger.info(f"Processed {len(processed_articles)} relevant articles")
        return processed_articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Alpha Vantage failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Alpha Vantage: {str(e)}")
        return None


def process_alpha_vantage_articles(articles, ticker, limit):
    """
    Process and format Alpha Vantage news articles
    """
    processed_articles = []
    seen_urls = set()
    
    for article in articles:
        # Skip duplicates
        url = article.get('url', '')
        if url in seen_urls or not url:
            continue
        seen_urls.add(url)
        
        # Extract basic info
        title = article.get('title', '')
        summary = article.get('summary', '')
        
        if not title or not summary:
            continue
            
        # Get sentiment information
        sentiment_data = article.get('overall_sentiment_label', 'Neutral')
        sentiment_score = article.get('overall_sentiment_score', 0)
        
        # Get ticker-specific sentiment if available
        ticker_sentiment = None
        ticker_relevance = 0
        
        ticker_sentiments = article.get('ticker_sentiment', [])
        for ts in ticker_sentiments:
            if ts.get('ticker', '').upper() == ticker.upper():
                ticker_sentiment = ts.get('sentiment_label', 'Neutral')
                ticker_relevance = float(ts.get('relevance_score', 0))
                break
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(title, summary, ticker, ticker_relevance)
        
        formatted_article = {
            "title": title,
            "url": url,
            "publishedAt": article.get('time_published', ''),
            "description": summary[:300] + "..." if len(summary) > 300 else summary,
            "source": article.get('source', 'Alpha Vantage'),
            "sentiment": ticker_sentiment or sentiment_data,
            "sentiment_score": sentiment_score,
            "relevance_score": relevance_score,
            "ticker": ticker.upper(),
            "authors": ", ".join(article.get('authors', [])) if article.get('authors') else 'Unknown',
            "category": article.get('category_within_source', 'Financial')
        }
        
        processed_articles.append(formatted_article)
        
        # Stop if we have enough articles
        if len(processed_articles) >= limit * 2:
            break
    
    # Sort by relevance and recency
    processed_articles.sort(key=lambda x: (x['relevance_score'], x['publishedAt']), reverse=True)
    
    return processed_articles[:limit]


def calculate_relevance_score(title, summary, ticker, ticker_relevance):
    """
    Calculate relevance score for an article
    """
    content = (title + ' ' + summary).lower()
    ticker_lower = ticker.lower()
    
    score = 0
    
    # Base score from Alpha Vantage relevance
    score += ticker_relevance * 10
    
    # Ticker mentions
    score += content.count(ticker_lower) * 5
    
    # Financial keywords
    financial_keywords = [
        'earnings', 'revenue', 'profit', 'stock', 'shares', 'market', 'trading',
        'investor', 'investment', 'analyst', 'forecast', 'dividend', 'growth',
        'decline', 'merger', 'acquisition', 'ipo', 'sec', 'filing'
    ]
    
    for keyword in financial_keywords:
        if keyword in content:
            score += 1
            
    return min(score, 50)  # Cap at 50


def get_fallback_news(ticker: str, limit: int = 5):
    """
    Provide fallback news data when API is unavailable
    """
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    fallback_articles = [
        {
            "title": f"{ticker.upper()} Stock Analysis: Technical and Fundamental Review",
            "url": f"https://finance.example.com/analysis/{ticker.lower()}-stock-review",
            "publishedAt": current_time,
            "description": f"Comprehensive analysis of {ticker.upper()} including technical indicators, fundamental metrics, and market position. Key support and resistance levels identified.",
            "source": "Financial Analysis Network",
            "sentiment": "Neutral",
            "sentiment_score": 0.1,
            "relevance_score": 10,
            "ticker": ticker.upper(),
            "authors": "Market Research Team",
            "category": "Analysis",
            "note": "Generated content - API unavailable"
        },
        {
            "title": f"{ticker.upper()} Quarterly Performance and Earnings Outlook",
            "url": f"https://finance.example.com/earnings/{ticker.lower()}-quarterly-outlook",
            "publishedAt": current_time,
            "description": f"Latest quarterly performance metrics for {ticker.upper()}, earnings expectations, and analyst consensus ratings. Revenue growth and margin analysis included.",
            "source": "Earnings Intelligence",
            "sentiment": "Positive",
            "sentiment_score": 0.3,
            "relevance_score": 9,
            "ticker": ticker.upper(),
            "authors": "Earnings Research Team",
            "category": "Earnings",
            "note": "Generated content - API unavailable"
        },
        {
            "title": f"{ticker.upper()} Market Trends and Trading Volume Analysis",
            "url": f"https://finance.example.com/trading/{ticker.lower()}-volume-trends",
            "publishedAt": current_time,
            "description": f"Recent trading patterns for {ticker.upper()} showing volume trends, price action, and institutional activity. Market sentiment indicators analyzed.",
            "source": "Trading Insights",
            "sentiment": "Neutral",
            "sentiment_score": 0.05,
            "relevance_score": 8,
            "ticker": ticker.upper(),
            "authors": "Trading Analytics Team",
            "category": "Trading",
            "note": "Generated content - API unavailable"
        },
        {
            "title": f"{ticker.upper()} Industry Position and Competitive Analysis",
            "url": f"https://finance.example.com/industry/{ticker.lower()}-competitive-position",
            "publishedAt": current_time,
            "description": f"Analysis of {ticker.upper()}'s position within its industry sector, competitive advantages, and market share dynamics. Growth prospects evaluated.",
            "source": "Industry Research",
            "sentiment": "Positive",
            "sentiment_score": 0.2,
            "relevance_score": 7,
            "ticker": ticker.upper(),
            "authors": "Sector Analysis Team",
            "category": "Industry",
            "note": "Generated content - API unavailable"
        },
        {
            "title": f"{ticker.upper()} Risk Assessment and Investment Considerations",
            "url": f"https://finance.example.com/risk/{ticker.lower()}-investment-risk",
            "publishedAt": current_time,
            "description": f"Risk profile analysis for {ticker.upper()} covering market risks, operational risks, and regulatory considerations. Investment recommendations provided.",
            "source": "Risk Management Weekly",
            "sentiment": "Neutral",
            "sentiment_score": -0.1,
            "relevance_score": 6,
            "ticker": ticker.upper(),
            "authors": "Risk Assessment Team",
            "category": "Risk Analysis",
            "note": "Generated content - API unavailable"
        }
    ]
    
    return fallback_articles[:limit]


def test_alpha_vantage_api():
    """
    Test function to verify Alpha Vantage API is working
    """
    print("🔍 Testing Alpha Vantage News API...")
    print(f"📊 Alpha Vantage API Key available: {'✅ Yes' if settings.ALPHA_VANTAGE_API_KEY else '❌ No'}")
    
    if settings.ALPHA_VANTAGE_API_KEY:
        print(f"🔑 API Key (first 8 chars): {settings.ALPHA_VANTAGE_API_KEY[:8]}...")
        
        # Test with popular stocks
        test_tickers = ["AAPL", "TSLA", "MSFT"]
        
        for ticker in test_tickers:
            print(f"\n📈 Testing {ticker}:")
            test_news = fetch_news(ticker, 3)
            print(f"  📰 Articles retrieved: {len(test_news)}")
            
            for i, article in enumerate(test_news[:2], 1):
                print(f"  {i}. {article.get('title', 'No title')[:60]}...")
                print(f"     💭 Sentiment: {article.get('sentiment', 'N/A')} | 📊 Score: {article.get('relevance_score', 0)}")
                if 'note' in article:
                    print(f"     ⚠️  {article['note']}")
    else:
        print("❌ No Alpha Vantage API key found - using fallback data")
        test_news = get_fallback_news("AAPL", 3)
        print(f"📰 Fallback results: {len(test_news)} articles")
        
    print("\n✅ Test completed!")


def get_news_summary(ticker: str):
    """
    Get a summary of recent news sentiment for a ticker
    """
    articles = fetch_news(ticker, 10)
    
    if not articles:
        return {"error": "No news data available"}
    
    sentiments = []
    total_relevance = 0
    
    for article in articles:
        if article.get('sentiment_score') is not None:
            sentiments.append(article.get('sentiment_score', 0))
        total_relevance += article.get('relevance_score', 0)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    return {
        "ticker": ticker.upper(),
        "total_articles": len(articles),
        "average_sentiment_score": round(avg_sentiment, 3),
        "total_relevance_score": total_relevance,
        "latest_headline": articles[0].get('title', '') if articles else '',
        "sentiment_label": "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral",
        "data_source": "Alpha Vantage" if settings.ALPHA_VANTAGE_API_KEY else "Fallback"
    }


def check_api_status():
    """
    Quick function to check if Alpha Vantage API is working
    """
    if not settings.ALPHA_VANTAGE_API_KEY:
        return {
            "status": "error",
            "message": "No Alpha Vantage API key configured",
            "recommendation": "Add ALPHA_VANTAGE_API_KEY to your .env file"
        }
    
    try:
        # Quick test API call
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": "AAPL",
            "apikey": settings.ALPHA_VANTAGE_API_KEY,
            "limit": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            return {
                "status": "error",
                "message": f"API Error: {data['Error Message']}",
                "recommendation": "Check your API key"
            }
        
        if "Note" in data:
            return {
                "status": "warning",
                "message": f"Rate Limited: {data['Note']}",
                "recommendation": "Wait a moment before trying again"
            }
        
        if "feed" in data and len(data["feed"]) > 0:
            return {
                "status": "success",
                "message": "Alpha Vantage API is working correctly",
                "articles_available": len(data["feed"])
            }
        
        return {
            "status": "warning",
            "message": "API working but no articles returned",
            "recommendation": "Try with different tickers"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}",
            "recommendation": "Check your internet connection"
        }


if __name__ == "__main__":
    # Run tests when file is executed directly
    print("🚀 Running Alpha Vantage News API Tests...\n")
    
    # Check API status first
    status = check_api_status()
    print(f"📋 API Status: {status['status'].upper()}")
    print(f"💬 Message: {status['message']}")
    if 'recommendation' in status:
        print(f"💡 Recommendation: {status['recommendation']}")
    
    print("\n" + "="*50 + "\n")
    
    # Run full test
    test_alpha_vantage_api()