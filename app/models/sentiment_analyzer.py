import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import yfinance as yf
from collections import defaultdict
import warnings
import os

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial markets"""
    
    def __init__(self, news_api_key: str = None, alpha_vantage_key: str = None):
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Sentiment scoring weights
        self.sentiment_weights = {
            'news_headlines': 0.4,
            'news_content': 0.3,
            'social_mentions': 0.2,
            'analyst_sentiment': 0.1
        }
        
        # Financial keywords for sentiment enhancement
        self.positive_keywords = [
            'bullish', 'upgrade', 'outperform', 'buy', 'strong', 'growth', 'profit',
            'earnings beat', 'revenue growth', 'expansion', 'acquisition', 'innovation',
            'partnership', 'dividend increase', 'share buyback', 'positive outlook'
        ]
        
        self.negative_keywords = [
            'bearish', 'downgrade', 'underperform', 'sell', 'weak', 'decline', 'loss',
            'earnings miss', 'revenue drop', 'layoffs', 'investigation', 'lawsuit',
            'bankruptcy', 'debt', 'recession', 'bearish outlook', 'guidance cut'
        ]
        
        # Volatility indicators
        self.volatility_keywords = [
            'volatile', 'uncertainty', 'risk', 'concern', 'worry', 'fear',
            'panic', 'crash', 'bubble', 'speculation', 'unstable'
        ]
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_enhanced_sentiment(self, text: str) -> Dict:
        """Calculate enhanced sentiment score with financial context"""
        
        cleaned_text = self._clean_text(text)
        
        # Basic TextBlob sentiment
        blob = TextBlob(cleaned_text)
        base_polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced scoring with financial keywords
        positive_score = sum(1 for keyword in self.positive_keywords if keyword in cleaned_text)
        negative_score = sum(1 for keyword in self.negative_keywords if keyword in cleaned_text)
        volatility_score = sum(1 for keyword in self.volatility_keywords if keyword in cleaned_text)
        
        # Adjust sentiment based on financial keywords
        keyword_adjustment = (positive_score - negative_score) * 0.1
        enhanced_polarity = np.clip(base_polarity + keyword_adjustment, -1, 1)
        
        # Confidence score based on keyword presence and subjectivity
        confidence = min(1.0, (positive_score + negative_score + volatility_score) * 0.2 + (1 - subjectivity))
        
        return {
            'polarity': float(enhanced_polarity),
            'subjectivity': float(subjectivity),
            'confidence': float(confidence),
            'positive_indicators': positive_score,
            'negative_indicators': negative_score,
            'volatility_indicators': volatility_score,
            'raw_polarity': float(base_polarity)
        }
    
    def get_news_sentiment(self, symbol: str, days_back: int = 7, max_articles: int = 100) -> Dict:
        """Fetch and analyze news sentiment for a given symbol"""
        
        if not self.news_api_key:
            return self._get_fallback_sentiment(symbol)
        
        try:
            # Fetch news articles
            articles = self._fetch_news_articles(symbol, days_back, max_articles)
            
            if not articles:
                return self._get_fallback_sentiment(symbol)
            
            # Analyze sentiment for each article
            sentiment_scores = []
            headline_sentiments = []
            content_sentiments = []
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                # Analyze headline sentiment
                if title:
                    headline_sentiment = self._calculate_enhanced_sentiment(title)
                    headline_sentiments.append(headline_sentiment)
                
                # Analyze content sentiment
                full_text = f"{title} {description} {content}"
                if full_text.strip():
                    content_sentiment = self._calculate_enhanced_sentiment(full_text)
                    content_sentiments.append(content_sentiment)
                    sentiment_scores.append(content_sentiment['polarity'])
            
            # Aggregate sentiment scores
            if not sentiment_scores:
                return self._get_fallback_sentiment(symbol)
            
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            # Calculate weighted sentiment
            headline_avg = np.mean([s['polarity'] for s in headline_sentiments]) if headline_sentiments else 0
            content_avg = np.mean([s['polarity'] for s in content_sentiments]) if content_sentiments else 0
            
            weighted_sentiment = (
                headline_avg * self.sentiment_weights['news_headlines'] +
                content_avg * self.sentiment_weights['news_content']
            )
            
            # Determine sentiment label
            sentiment_label = self._get_sentiment_label(weighted_sentiment)
            
            # Calculate confidence based on article count and consistency
            confidence = min(1.0, len(articles) / 20) * (1 - min(sentiment_std, 0.5) / 0.5)
            
            # Count sentiment distribution
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            return {
                'symbol': symbol,
                'sentiment_score': float(weighted_sentiment),
                'sentiment_label': sentiment_label,
                'confidence': float(confidence),
                'article_count': len(articles),
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'sentiment_std': float(sentiment_std),
                'headline_sentiment': float(headline_avg),
                'content_sentiment': float(content_avg),
                'days_analyzed': days_back,
                'analysis_date': datetime.now().isoformat(),
                'top_keywords': self._extract_top_keywords(articles)
            }
            
        except Exception as e:
            print(f"Error analyzing news sentiment for {symbol}: {str(e)}")
            return self._get_fallback_sentiment(symbol)
    
    def _fetch_news_articles(self, symbol: str, days_back: int, max_articles: int) -> List[Dict]:
        """Fetch news articles from News API"""
        
        try:
            # Get company info for better search terms
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            # Search terms
            search_terms = f'"{symbol}" OR "{company_name}" stock earnings financial'
            
            # API parameters
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_terms,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': min(max_articles, 100),
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                print(f"News API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news articles: {str(e)}")
            return []
    
    def _get_fallback_sentiment(self, symbol: str) -> Dict:
        """Provide fallback sentiment analysis when APIs are unavailable"""
        
        try:
            # Use recent price action as proxy for sentiment
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="30d")
            
            if df.empty:
                return self._get_neutral_sentiment(symbol)
            
            # Calculate price momentum
            recent_return = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
            monthly_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            
            # Volume analysis
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Simple sentiment estimation
            price_sentiment = np.tanh(recent_return * 10)  # Scale to [-1, 1]
            volume_boost = min(volume_ratio / 2, 0.5)  # Volume can boost sentiment
            
            estimated_sentiment = np.clip(price_sentiment + volume_boost - 0.25, -1, 1)
            
            return {
                'symbol': symbol,
                'sentiment_score': float(estimated_sentiment),
                'sentiment_label': self._get_sentiment_label(estimated_sentiment),
                'confidence': 0.3,  # Low confidence for fallback
                'article_count': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'data_source': 'price_action_proxy',
                'recent_return': float(recent_return),
                'monthly_return': float(monthly_return),
                'analysis_date': datetime.now().isoformat(),
                'note': 'Sentiment estimated from price action due to limited news data'
            }
            
        except Exception as e:
            print(f"Error in fallback sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment(symbol)
    
    def _get_neutral_sentiment(self, symbol: str) -> Dict:
        """Return neutral sentiment when analysis fails"""
        return {
            'symbol': symbol,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.1,
            'article_count': 0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 1},
            'data_source': 'default_neutral',
            'analysis_date': datetime.now().isoformat(),
            'note': 'No sentiment data available - returning neutral default'
        }
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to human-readable label"""
        if sentiment_score >= 0.2:
            return 'positive'
        elif sentiment_score <= -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_top_keywords(self, articles: List[Dict], top_n: int = 10) -> List[str]:
        """Extract top keywords from article content"""
        
        try:
            word_freq = defaultdict(int)
            
            # Common words to ignore
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'said', 'says', 'this', 'that', 'these', 'those'}
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                
                text = f"{title} {description}".lower()
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                
                for word in words:
                    if word not in stop_words and len(word) > 2:
                        word_freq[word] += 1
            
            # Return top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [keyword for keyword, _ in top_keywords]
            
        except Exception as e:
            print(f"Error extracting keywords: {str(e)}")
            return []
    
    def analyze_market_sentiment(self, symbols: List[str], sector: str = None) -> Dict:
        """Analyze overall market sentiment for a group of symbols"""
        
        sector_sentiments = {}
        overall_scores = []
        
        for symbol in symbols:
            sentiment_data = self.get_news_sentiment(symbol, days_back=7)
            sector_sentiments[symbol] = sentiment_data
            
            if sentiment_data['confidence'] > 0.2:  # Only include confident predictions
                overall_scores.append(sentiment_data['sentiment_score'])
        
        if not overall_scores:
            overall_sentiment = 0.0
            overall_label = 'neutral'
            confidence = 0.1
        else:
            overall_sentiment = np.mean(overall_scores)
            overall_label = self._get_sentiment_label(overall_sentiment)
            confidence = min(1.0, len(overall_scores) / len(symbols))
        
        # Calculate sentiment distribution
        positive_count = sum(1 for score in overall_scores if score > 0.1)
        negative_count = sum(1 for score in overall_scores if score < -0.1)
        neutral_count = len(overall_scores) - positive_count - negative_count
        
        return {
            'sector': sector or 'mixed',
            'overall_sentiment': float(overall_sentiment),
            'overall_label': overall_label,
            'confidence': float(confidence),
            'symbols_analyzed': len(symbols),
            'valid_analyses': len(overall_scores),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'individual_sentiments': sector_sentiments,
            'analysis_date': datetime.now().isoformat()
        }
    
    def get_sentiment_trend(self, symbol: str, days: int = 30) -> Dict:
        """Analyze sentiment trend over time"""
        
        # This is a simplified implementation
        # In production, you'd want to store historical sentiment data
        
        try:
            # Analyze different time windows
            short_term = self.get_news_sentiment(symbol, days_back=7)
            medium_term = self.get_news_sentiment(symbol, days_back=14)
            long_term = self.get_news_sentiment(symbol, days_back=30)
            
            # Calculate trend
            if (short_term['confidence'] > 0.2 and medium_term['confidence'] > 0.2 and 
                long_term['confidence'] > 0.2):
                
                trend_direction = 'improving' if short_term['sentiment_score'] > long_term['sentiment_score'] else 'declining'
                trend_strength = abs(short_term['sentiment_score'] - long_term['sentiment_score'])
                
            else:
                trend_direction = 'unknown'
                trend_strength = 0.0
            
            return {
                'symbol': symbol,
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'short_term_sentiment': short_term,
                'medium_term_sentiment': medium_term,
                'long_term_sentiment': long_term,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment trend for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'trend_direction': 'unknown',
                'trend_strength': 0.0,
                'error': str(e)
            }
    
    def sentiment_impact_on_prediction(self, sentiment_data: Dict, base_prediction: float, 
                                     current_price: float) -> Dict:
        """Adjust predictions based on sentiment analysis"""
        
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        confidence = sentiment_data.get('confidence', 0.1)
        
        # Calculate sentiment adjustment factor
        max_adjustment = 0.05  # Maximum 5% adjustment
        sentiment_adjustment = sentiment_score * max_adjustment * confidence
        
        # Apply adjustment to prediction
        adjusted_prediction = base_prediction * (1 + sentiment_adjustment)
        
        # Calculate impact metrics
        prediction_change = adjusted_prediction - base_prediction
        prediction_change_pct = (prediction_change / base_prediction) * 100 if base_prediction != 0 else 0
        
        return {
            'original_prediction': float(base_prediction),
            'adjusted_prediction': float(adjusted_prediction),
            'sentiment_adjustment': float(sentiment_adjustment),
            'prediction_change': float(prediction_change),
            'prediction_change_pct': float(prediction_change_pct),
            'sentiment_confidence': float(confidence),
            'sentiment_score': float(sentiment_score),
            'adjustment_rationale': self._get_adjustment_rationale(sentiment_data)
        }
    
    def _get_adjustment_rationale(self, sentiment_data: Dict) -> str:
        """Generate explanation for sentiment-based prediction adjustment"""
        
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
        confidence = sentiment_data.get('confidence', 0)
        article_count = sentiment_data.get('article_count', 0)
        
        if confidence < 0.3:
            return f"Low confidence sentiment analysis ({confidence:.1%}) - minimal adjustment applied."
        
        if sentiment_label == 'positive':
            return f"Positive news sentiment ({sentiment_score:.2f}) from {article_count} articles suggests upward price pressure."
        elif sentiment_label == 'negative':
            return f"Negative news sentiment ({sentiment_score:.2f}) from {article_count} articles suggests downward price pressure."
        else:
            return f"Neutral sentiment ({sentiment_score:.2f}) - no significant adjustment warranted."