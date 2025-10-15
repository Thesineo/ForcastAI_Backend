"""
Explanation Service for AI Financial Advisor
AI-powered explanations, insights, and educational content generation
Integrates all services to provide comprehensive investment analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from functools import lru_cache
import time
import numpy as np

# Import our services
from app.services.cache import CacheService, CacheType, get_cache_service
from app.services.market_data_services import MarketDataService, StockQuote, CompanyInfo, get_market_service
from app.services.prediction_services import PredictionService, PredictionResult, TradingSignal, SignalType, get_prediction_service
from app.services.news_services import NewsService, SentimentAnalysis, NewsArticle, get_news_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations the service can generate"""
    STOCK_ANALYSIS = "stock_analysis"
    TRADING_SIGNAL = "trading_signal" 
    PRICE_MOVEMENT = "price_movement"
    MARKET_OVERVIEW = "market_overview"
    RISK_ASSESSMENT = "risk_assessment"
    EDUCATIONAL = "educational"
    PORTFOLIO_REVIEW = "portfolio_review"
    NEWS_IMPACT = "news_impact"

class ComplexityLevel(Enum):
    """Explanation complexity levels"""
    BEGINNER = "beginner"       # Simple, educational explanations
    INTERMEDIATE = "intermediate"  # Moderate detail with some technical terms
    ADVANCED = "advanced"       # Full technical analysis with all metrics

class ToneStyle(Enum):
    """Explanation tone and style"""
    PROFESSIONAL = "professional"  # Formal financial advisor tone
    CONVERSATIONAL = "conversational"  # Friendly, approachable tone  
    EDUCATIONAL = "educational"    # Teaching-focused explanations
    CONFIDENT = "confident"       # Assertive recommendations
    CAUTIOUS = "cautious"         # Risk-focused, conservative tone

@dataclass
class ExplanationContext:
    """Context for generating explanations"""
    user_experience_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    preferred_tone: ToneStyle = ToneStyle.CONVERSATIONAL
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    investment_timeframe: str = "medium_term"  # short_term, medium_term, long_term
    focus_areas: List[str] = None  # technical, fundamental, news, risk
    include_educational: bool = True
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = ["technical", "fundamental", "news"]

@dataclass
class Insight:
    """Individual insight or key point"""
    category: str  # technical, fundamental, news, risk, opportunity
    title: str
    description: str
    importance: float  # 0-1 scale
    confidence: float  # 0-1 scale
    supporting_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.supporting_data is None:
            self.supporting_data = {}

@dataclass
class Explanation:
    """Complete explanation with insights and recommendations"""
    explanation_type: ExplanationType
    symbol: str
    title: str
    summary: str
    key_insights: List[Insight]
    detailed_analysis: str
    recommendations: List[str]
    risk_warnings: List[str]
    educational_notes: List[str]
    confidence_score: float
    data_sources: List[str]
    methodology: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class ExplanationService:
    """AI-powered explanation and insight generation service"""
    
    # Template phrases for different tones
    TONE_TEMPLATES = {
        ToneStyle.PROFESSIONAL: {
            "intro": "Based on our comprehensive analysis,",
            "recommendation": "We recommend",
            "caution": "Please note that",
            "conclusion": "In conclusion,"
        },
        ToneStyle.CONVERSATIONAL: {
            "intro": "Here's what I found when analyzing",
            "recommendation": "I'd suggest",
            "caution": "Keep in mind that",
            "conclusion": "To sum it up,"
        },
        ToneStyle.EDUCATIONAL: {
            "intro": "Let me explain what's happening with",
            "recommendation": "Based on this analysis, you might consider",
            "caution": "It's important to understand that",
            "conclusion": "What this means for you is"
        },
        ToneStyle.CONFIDENT: {
            "intro": "The data clearly shows that",
            "recommendation": "I strongly recommend",
            "caution": "Be aware that",
            "conclusion": "The bottom line is"
        },
        ToneStyle.CAUTIOUS: {
            "intro": "While analyzing the available data,",
            "recommendation": "You may want to consider",
            "caution": "However, it's crucial to note that",
            "conclusion": "Given these considerations,"
        }
    }
    
    # Educational explanations for technical terms
    EDUCATIONAL_TERMS = {
        "RSI": "RSI (Relative Strength Index) measures if a stock is overbought (>70) or oversold (<30). It helps identify potential reversal points.",
        "MACD": "MACD shows the relationship between two moving averages. When MACD crosses above its signal line, it often indicates upward momentum.",
        "Support": "Support is a price level where a stock tends to find buying interest and bounce back up.",
        "Resistance": "Resistance is a price level where a stock faces selling pressure and has difficulty breaking through.",
        "Volume": "Trading volume shows investor interest. High volume during price moves suggests stronger conviction.",
        "Volatility": "Volatility measures how much a stock's price fluctuates. Higher volatility means more risk but also more potential reward.",
        "Beta": "Beta measures how much a stock moves relative to the overall market. A beta above 1 means it's more volatile than the market.",
        "P/E Ratio": "Price-to-Earnings ratio compares a stock's price to its earnings per share. Lower P/E might indicate value.",
        "Market Cap": "Market capitalization is the total value of all shares. It determines if a company is small, mid, or large-cap."
    }
    
    def __init__(self, 
                 cache_service: CacheService = None,
                 market_service: MarketDataService = None,
                 prediction_service: PredictionService = None,
                 news_service: NewsService = None):
        
        self.cache = cache_service or get_cache_service()
        self.market_service = market_service or get_market_service()
        self.prediction_service = prediction_service or get_prediction_service()
        self.news_service = news_service or get_news_service()
        
        logger.info("Explanation service initialized with all components")
    
    async def explain_stock_analysis(self, 
                                   symbol: str,
                                   context: ExplanationContext = None) -> Optional[Explanation]:
        """Generate comprehensive stock analysis explanation"""
        
        if context is None:
            context = ExplanationContext()
        
        start_time = time.time()
        
        try:
            # Gather all data
            logger.info(f"Generating stock analysis explanation for {symbol}")
            
            # Get market data
            quote = self.market_service.get_stock_quote(symbol)
            company_info = self.market_service.get_company_info(symbol)
            historical_data = self.market_service.get_historical_data(symbol, "1y", "1d")
            
            if not quote:
                logger.error(f"No quote data available for {symbol}")
                return None
            
            # Get predictions and signals
            df = self.market_service._dict_to_df(historical_data) if not historical_data.get("empty") else None
            prediction = None
            signal = None
            
            if df is not None and not df.empty:
                prediction = self.prediction_service.predict_stock_price(df, symbol)
                if prediction:
                    signal = self.prediction_service.generate_trading_signal(df, symbol, prediction)
            
            # Get news sentiment
            sentiment_analysis = await self.news_service.analyze_sentiment(symbol, hours_back=48)
            
            # Generate insights
            insights = self._generate_stock_insights(
                quote, company_info, prediction, signal, sentiment_analysis, context
            )
            
            # Create explanation
            explanation = self._build_stock_explanation(
                symbol, quote, company_info, prediction, signal, 
                sentiment_analysis, insights, context
            )
            
            # Cache the explanation
            cache_key = f"{symbol}_analysis"
            self.cache.set(CacheType.ANALYSIS, cache_key, asdict(explanation), ttl=1800)
            
            execution_time = time.time() - start_time
            logger.info(f"Generated stock analysis for {symbol} in {execution_time:.2f}s")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating stock analysis for {symbol}: {e}")
            return None
    
    def _generate_stock_insights(self,
                               quote: StockQuote,
                               company_info: Optional[CompanyInfo],
                               prediction: Optional[PredictionResult],
                               signal: Optional[TradingSignal],
                               sentiment: Optional[SentimentAnalysis],
                               context: ExplanationContext) -> List[Insight]:
        """Generate key insights for stock analysis"""
        
        insights = []
        
        try:
            # Price movement insight
            if abs(quote.change_percent) > 2:
                insights.append(Insight(
                    category="technical",
                    title=f"Significant Price Movement",
                    description=f"{quote.symbol} moved {quote.change_percent:+.2f}% today to ${quote.price:.2f}. " +
                               f"This {'gain' if quote.change > 0 else 'decline'} is notable as it exceeds typical daily volatility.",
                    importance=0.9,
                    confidence=1.0,
                    supporting_data={"change_percent": quote.change_percent, "price": quote.price}
                ))
            
            # Technical analysis insights
            if prediction and prediction.technical_indicators:
                indicators = prediction.technical_indicators
                
                # RSI insight
                if indicators.rsi < 30:
                    insights.append(Insight(
                        category="technical",
                        title="Oversold Condition Detected",
                        description=f"RSI is at {indicators.rsi:.1f}, indicating {quote.symbol} may be oversold. " +
                                   "This could present a buying opportunity if other factors align.",
                        importance=0.8,
                        confidence=0.7,
                        supporting_data={"rsi": indicators.rsi, "threshold": 30}
                    ))
                elif indicators.rsi > 70:
                    insights.append(Insight(
                        category="technical",
                        title="Overbought Condition Warning",
                        description=f"RSI is at {indicators.rsi:.1f}, suggesting {quote.symbol} may be overbought. " +
                                   "Consider waiting for a pullback before buying.",
                        importance=0.8,
                        confidence=0.7,
                        supporting_data={"rsi": indicators.rsi, "threshold": 70}
                    ))
                
                # Moving average insight
                if quote.price > indicators.sma_20 > indicators.sma_50:
                    insights.append(Insight(
                        category="technical",
                        title="Positive Trend Structure",
                        description=f"Price (${quote.price:.2f}) is above both 20-day (${indicators.sma_20:.2f}) " +
                                   f"and 50-day (${indicators.sma_50:.2f}) moving averages, indicating an upward trend.",
                        importance=0.7,
                        confidence=0.8,
                        supporting_data={
                            "price": quote.price,
                            "sma_20": indicators.sma_20,
                            "sma_50": indicators.sma_50
                        }
                    ))
                elif quote.price < indicators.sma_20 < indicators.sma_50:
                    insights.append(Insight(
                        category="technical",
                        title="Concerning Trend Pattern",
                        description=f"Price (${quote.price:.2f}) is below both moving averages, " +
                                   "suggesting potential downward pressure.",
                        importance=0.7,
                        confidence=0.8,
                        supporting_data={
                            "price": quote.price,
                            "sma_20": indicators.sma_20,
                            "sma_50": indicators.sma_50
                        }
                    ))
            
            # Prediction insight
            if prediction:
                price_change = prediction.price_change_percent
                if abs(price_change) > 2:
                    direction = "upward" if price_change > 0 else "downward"
                    insights.append(Insight(
                        category="technical",
                        title=f"Prediction Shows {direction.title()} Movement",
                        description=f"Our {prediction.model_used} model predicts {quote.symbol} could move " +
                                   f"{price_change:+.1f}% to ${prediction.predicted_price:.2f} " +
                                   f"with {prediction.confidence_score:.0%} confidence.",
                        importance=0.8,
                        confidence=prediction.confidence_score,
                        supporting_data={
                            "predicted_price": prediction.predicted_price,
                            "change_percent": price_change,
                            "model": prediction.model_used
                        }
                    ))
            
            # Trading signal insight
            if signal and signal.signal != SignalType.HOLD:
                action_word = "buying" if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else "selling"
                strength = "strong " if "STRONG" in signal.signal.value else ""
                
                insights.append(Insight(
                    category="technical",
                    title=f"{strength.title()}{signal.signal.value.replace('_', ' ').title()} Signal Generated",
                    description=f"Technical analysis suggests {strength}{action_word} {quote.symbol} " +
                               f"with {signal.confidence:.0%} confidence. " +
                               f"Key factors: {', '.join(signal.reasoning[:2])}.",
                    importance=0.9,
                    confidence=signal.confidence,
                    supporting_data={
                        "signal": signal.signal.value,
                        "reasoning": signal.reasoning,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss
                    }
                ))
            
            # Fundamental insights
            if company_info:
                # Market cap insight
                if company_info.market_cap:
                    market_cap_b = company_info.market_cap / 1e9
                    size_category = "large" if market_cap_b > 10 else "mid" if market_cap_b > 2 else "small"
                    
                    insights.append(Insight(
                        category="fundamental",
                        title=f"{size_category.title()}-Cap Company Analysis",
                        description=f"{quote.symbol} has a market cap of ${market_cap_b:.1f}B, " +
                                   f"classifying it as a {size_category}-cap stock. " +
                                   f"This typically means {'higher stability but slower growth' if size_category == 'large' else 'higher growth potential but more volatility' if size_category == 'small' else 'balanced risk-reward profile'}.",
                        importance=0.6,
                        confidence=0.9,
                        supporting_data={"market_cap": company_info.market_cap, "category": size_category}
                    ))
                
                # P/E ratio insight
                if company_info.pe_ratio and company_info.pe_ratio > 0:
                    pe_assessment = "high" if company_info.pe_ratio > 25 else "low" if company_info.pe_ratio < 15 else "moderate"
                    insights.append(Insight(
                        category="fundamental",
                        title=f"Valuation Assessment: {pe_assessment.title()} P/E Ratio",
                        description=f"P/E ratio of {company_info.pe_ratio:.1f} is considered {pe_assessment} " +
                                   f"{'suggesting the stock may be expensive' if pe_assessment == 'high' else 'indicating potential value opportunity' if pe_assessment == 'low' else 'showing reasonable valuation'}.",
                        importance=0.7,
                        confidence=0.8,
                        supporting_data={"pe_ratio": company_info.pe_ratio, "assessment": pe_assessment}
                    ))
            
            # News sentiment insights
            if sentiment:
                sentiment_strength = "very " if "VERY" in sentiment.overall_sentiment.value.upper() else ""
                sentiment_direction = sentiment.overall_sentiment.value.lower().replace("very ", "")
                
                if sentiment.total_articles >= 5:  # Only if sufficient news volume
                    insights.append(Insight(
                        category="news",
                        title=f"{sentiment_strength.title()}{sentiment_direction.title()} News Sentiment",
                        description=f"Analysis of {sentiment.total_articles} recent articles shows " +
                                   f"{sentiment_strength}{sentiment_direction} sentiment " +
                                   f"(score: {sentiment.sentiment_score:+.2f}). " +
                                   f"Trending topics: {', '.join(sentiment.trending_keywords[:3])}.",
                        importance=0.7,
                        confidence=sentiment.confidence,
                        supporting_data={
                            "sentiment_score": sentiment.sentiment_score,
                            "total_articles": sentiment.total_articles,
                            "keywords": sentiment.trending_keywords
                        }
                    ))
                
                # Breaking news impact
                if sentiment.recent_breaking_news:
                    insights.append(Insight(
                        category="news",
                        title="Breaking News Impact",
                        description=f"Recent breaking news may be affecting {quote.symbol}. " +
                                   f"{len(sentiment.recent_breaking_news)} urgent articles detected. " +
                                   "Monitor for increased volatility.",
                        importance=0.9,
                        confidence=0.8,
                        supporting_data={"breaking_news_count": len(sentiment.recent_breaking_news)}
                    ))
            
            # Risk insights
            if prediction:
                volatility = prediction.volatility_score
                risk_level = "high" if volatility > 0.3 else "low" if volatility < 0.15 else "moderate"
                
                insights.append(Insight(
                    category="risk",
                    title=f"{risk_level.title()} Volatility Assessment",
                    description=f"Current volatility score of {volatility:.2f} indicates {risk_level} risk. " +
                               f"{'Consider position sizing carefully' if risk_level == 'high' else 'Standard risk management applies' if risk_level == 'moderate' else 'Relatively stable price movement expected'}.",
                    importance=0.6,
                    confidence=0.8,
                    supporting_data={"volatility_score": volatility, "risk_level": risk_level}
                ))
            
            # Sort insights by importance
            insights.sort(key=lambda x: x.importance, reverse=True)
            
            return insights[:8]  # Return top 8 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return insights
    
    def _build_stock_explanation(self,
                               symbol: str,
                               quote: StockQuote,
                               company_info: Optional[CompanyInfo],
                               prediction: Optional[PredictionResult],
                               signal: Optional[TradingSignal],
                               sentiment: Optional[SentimentAnalysis],
                               insights: List[Insight],
                               context: ExplanationContext) -> Explanation:
        """Build comprehensive stock explanation"""
        
        try:
            # Get tone templates
            templates = self.TONE_TEMPLATES[context.preferred_tone]
            
            # Generate title
            title = f"{symbol} Investment Analysis"
            
            # Generate summary
            summary = self._generate_summary(symbol, quote, prediction, signal, sentiment, templates, context)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                symbol, quote, company_info, prediction, signal, sentiment, insights, templates, context
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                symbol, quote, prediction, signal, sentiment, templates, context
            )
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(
                symbol, quote, prediction, signal, sentiment, context
            )
            
            # Generate educational notes
            educational_notes = self._generate_educational_notes(insights, context)
            
            # Calculate overall confidence
            confidence_score = self._calculate_explanation_confidence(
                quote, prediction, signal, sentiment, insights
            )
            
            # Data sources
            data_sources = ["Market Data", "Technical Analysis"]
            if sentiment:
                data_sources.append("News Analysis")
            if company_info:
                data_sources.append("Company Fundamentals")
            
            return Explanation(
                explanation_type=ExplanationType.STOCK_ANALYSIS,
                symbol=symbol,
                title=title,
                summary=summary,
                key_insights=insights,
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                educational_notes=educational_notes,
                confidence_score=confidence_score,
                data_sources=data_sources,
                methodology="Multi-factor analysis combining technical, fundamental, and sentiment data"
            )
            
        except Exception as e:
            logger.error(f"Error building explanation: {e}")
            # Return basic explanation as fallback
            return Explanation(
                explanation_type=ExplanationType.STOCK_ANALYSIS,
                symbol=symbol,
                title=f"{symbol} Analysis",
                summary=f"Analysis of {symbol} at ${quote.price:.2f} ({quote.change_percent:+.1f}%)",
                key_insights=insights,
                detailed_analysis="Analysis data temporarily unavailable.",
                recommendations=["Monitor price action", "Consider your risk tolerance"],
                risk_warnings=["All investments carry risk"],
                educational_notes=[],
                confidence_score=0.5,
                data_sources=["Market Data"],
                methodology="Basic analysis"
            )
    
    def _generate_summary(self,
                        symbol: str,
                        quote: StockQuote,
                        prediction: Optional[PredictionResult],
                        signal: Optional[TradingSignal],
                        sentiment: Optional[SentimentAnalysis],
                        templates: Dict[str, str],
                        context: ExplanationContext) -> str:
        """Generate executive summary"""
        
        try:
            # Start with current price and movement
            movement = "gained" if quote.change > 0 else "declined" if quote.change < 0 else "remained flat"
            
            summary_parts = [
                f"{templates['intro']} {symbol} is currently trading at ${quote.price:.2f}, "
                f"having {movement} {abs(quote.change_percent):.1f}% {'today' if quote.change != 0 else 'in recent trading'}."
            ]
            
            # Add prediction insight if available
            if prediction:
                direction = "upward" if prediction.price_change_percent > 0 else "downward"
                summary_parts.append(
                    f"Our analysis suggests potential {direction} movement to "
                    f"${prediction.predicted_price:.2f} ({prediction.price_change_percent:+.1f}%)."
                )
            
            # Add signal insight
            if signal and signal.signal != SignalType.HOLD:
                action = "buying opportunity" if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else "selling consideration"
                summary_parts.append(f"Technical indicators suggest this may be a {action}.")
            
            # Add sentiment context
            if sentiment and sentiment.total_articles >= 3:
                sent_desc = sentiment.overall_sentiment.value.lower()
                summary_parts.append(
                    f"Recent news sentiment appears {sent_desc} based on {sentiment.total_articles} articles analyzed."
                )
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Analysis of {symbol} at ${quote.price:.2f}."
    
    def _generate_detailed_analysis(self,
                                  symbol: str,
                                  quote: StockQuote,
                                  company_info: Optional[CompanyInfo],
                                  prediction: Optional[PredictionResult],
                                  signal: Optional[TradingSignal],
                                  sentiment: Optional[SentimentAnalysis],
                                  insights: List[Insight],
                                  templates: Dict[str, str],
                                  context: ExplanationContext) -> str:
        """Generate detailed analysis narrative"""
        
        try:
            analysis_sections = []
            
            # Technical Analysis Section
            if prediction and prediction.technical_indicators:
                tech_section = self._generate_technical_section(
                    symbol, quote, prediction, signal, templates, context
                )
                analysis_sections.append(tech_section)
            
            # Fundamental Analysis Section
            if company_info:
                fund_section = self._generate_fundamental_section(
                    symbol, company_info, templates, context
                )
                analysis_sections.append(fund_section)
            
            # News and Sentiment Section
            if sentiment:
                news_section = self._generate_news_section(
                    symbol, sentiment, templates, context
                )
                analysis_sections.append(news_section)
            
            # Key Insights Summary
            if insights:
                insights_section = self._generate_insights_section(insights, templates, context)
                analysis_sections.append(insights_section)
            
            return "\n\n".join(analysis_sections)
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            return f"Technical and fundamental analysis of {symbol} indicates mixed signals. Monitor key levels and news developments."
    
    def _generate_technical_section(self,
                                  symbol: str,
                                  quote: StockQuote,
                                  prediction: PredictionResult,
                                  signal: Optional[TradingSignal],
                                  templates: Dict[str, str],
                                  context: ExplanationContext) -> str:
        """Generate technical analysis section"""
        
        indicators = prediction.technical_indicators
        tech_parts = [f"**Technical Analysis:**"]
        
        # Price action
        tech_parts.append(
            f"At ${quote.price:.2f}, {symbol} shows "
            f"{'bullish' if quote.change_percent > 0 else 'bearish' if quote.change_percent < 0 else 'neutral'} "
            f"price action with {abs(quote.change_percent):.1f}% movement."
        )
        
        # RSI analysis
        if indicators.rsi:
            rsi_condition = "oversold" if indicators.rsi < 30 else "overbought" if indicators.rsi > 70 else "neutral"
            tech_parts.append(
                f"The RSI at {indicators.rsi:.1f} indicates {rsi_condition} conditions" +
                (f" - {self.EDUCATIONAL_TERMS.get('RSI', '')}" if context.include_educational else "") + "."
            )
        
        # Moving averages
        if indicators.sma_20 and indicators.sma_50:
            ma_trend = "upward" if quote.price > indicators.sma_20 > indicators.sma_50 else "downward" if quote.price < indicators.sma_20 < indicators.sma_50 else "sideways"
            tech_parts.append(
                f"Moving averages suggest a {ma_trend} trend with price "
                f"{'above' if quote.price > indicators.sma_20 else 'below'} the 20-day average (${indicators.sma_20:.2f})."
            )
        
        # MACD
        if indicators.macd and indicators.macd_signal:
            macd_signal = "bullish" if indicators.macd > indicators.macd_signal else "bearish"
            tech_parts.append(
                f"MACD shows {macd_signal} momentum" +
                (f" - {self.EDUCATIONAL_TERMS.get('MACD', '')}" if context.include_educational else "") + "."
            )
        
        # Support and resistance
        if prediction.support_levels and prediction.resistance_levels:
            tech_parts.append(
                f"Key levels to watch: support near ${prediction.support_levels[0]:.2f} "
                f"and resistance around ${prediction.resistance_levels[0]:.2f}."
            )
        
        return " ".join(tech_parts)
    
    def _generate_fundamental_section(self,
                                    symbol: str,
                                    company_info: CompanyInfo,
                                    templates: Dict[str, str],
                                    context: ExplanationContext) -> str:
        """Generate fundamental analysis section"""
        try:
            fund_parts = [f"**Fundamental Analysis:**"]
            
            # Company overview
            fund_parts.append(
                f"{symbol} operates in the {company_info.industry} sector"
                + (f" within {company_info.sector}" if company_info.sector != company_info.industry else "") + "."
            )
            
            # Market cap
            if company_info.market_cap:
                market_cap_b = company_info.market_cap / 1e9
                size = "large" if market_cap_b > 10 else "mid" if market_cap_b > 2 else "small"
                fund_parts.append(
                    f"With a market cap of ${market_cap_b:.1f}B, it's classified as a {size}-cap company."
                )
            
            # Valuation metrics
            valuation_parts = []
            if company_info.pe_ratio and company_info.pe_ratio > 0:
                pe_level = "high" if company_info.pe_ratio > 25 else "low" if company_info.pe_ratio < 15 else "moderate"
                valuation_parts.append(f"P/E ratio of {company_info.pe_ratio:.1f} ({pe_level})")
            
            if company_info.pb_ratio and company_info.pb_ratio > 0:
                valuation_parts.append(f"P/B ratio of {company_info.pb_ratio:.1f}")
            
            if valuation_parts:
                fund_parts.append(f"Valuation metrics show {', '.join(valuation_parts)}.")
            
            # Financial health
            if company_info.debt_to_equity is not None:
                debt_level = "high" if company_info.debt_to_equity > 50 else "low" if company_info.debt_to_equity < 25 else "moderate"
                fund_parts.append(f"Debt-to-equity ratio of {company_info.debt_to_equity:.1f} indicates {debt_level} leverage.")
            
            if company_info.roe is not None:
                roe_quality = "excellent" if company_info.roe > 20 else "good" if company_info.roe > 15 else "moderate" if company_info.roe > 10 else "concerning"
                fund_parts.append(f"Return on equity of {company_info.roe:.1f}% suggests {roe_quality} profitability.")
    
                # Risk assessment
                # (No signal context available here, so skip signal-based risk assessment and implementation guidance.)
        
            return " ".join(fund_parts)
        
        except Exception as e:
            logger.error(f"Error generating fundamental analysis: {e}")
            return "Fundamental analysis data temporarily unavailable."
    
    async def generate_portfolio_explanation(self,
                                           user_id: str,
                                           portfolio_data: Dict[str, Any],
                                           context: ExplanationContext = None) -> Optional[Explanation]:
        """Generate comprehensive portfolio analysis explanation"""
        
        if context is None:
            context = ExplanationContext()
        
        try:
            templates = self.TONE_TEMPLATES[context.preferred_tone]
            
            # Extract portfolio metrics
            total_value = portfolio_data.get('total_value', 0)
            day_change = portfolio_data.get('day_change', 0)
            day_change_percent = portfolio_data.get('day_change_percent', 0)
            positions = portfolio_data.get('positions', {})
            performance = portfolio_data.get('performance', {})
            risk_metrics = portfolio_data.get('risk_metrics', {})
            
            # Generate portfolio insights
            portfolio_insights = self._generate_portfolio_insights(
                total_value, day_change, day_change_percent, positions, performance, risk_metrics, context
            )
            
            # Create summary
            performance_desc = "gained" if day_change > 0 else "declined" if day_change < 0 else "remained stable"
            summary = (f"{templates['intro']} your portfolio is currently valued at ${total_value:,.2f}, "
                      f"having {performance_desc} ${abs(day_change):,.2f} "
                      f"({day_change_percent:+.2f}%) today. ")
            
            if len(positions) > 0:
                summary += f"Your {len(positions)} positions show {'mixed' if abs(day_change_percent) < 1 else 'strong' if day_change_percent > 2 else 'concerning' if day_change_percent < -2 else 'moderate'} performance."
            
            # Generate detailed analysis
            detailed_analysis = self._generate_portfolio_detailed_analysis(
                total_value, positions, performance, risk_metrics, templates, context
            )
            
            # Generate portfolio recommendations
            recommendations = self._generate_portfolio_recommendations(
                positions, performance, risk_metrics, templates, context
            )
            
            # Generate portfolio risk warnings
            risk_warnings = self._generate_portfolio_risk_warnings(
                positions, risk_metrics, context
            )
            
            # Educational content for portfolio management
            educational_notes = []
            if context.include_educational:
                educational_notes.extend([
                    "**Diversification:** Spreading investments across different stocks and sectors reduces risk.",
                    "**Rebalancing:** Periodically adjusting position sizes helps maintain target allocations.",
                    "**Risk Management:** Never put more than 5-10% in any single position unless it's part of a specific strategy."
                ])
            
            return Explanation(
                explanation_type=ExplanationType.PORTFOLIO_REVIEW,
                symbol="PORTFOLIO",
                title="Portfolio Performance Analysis",
                summary=summary,
                key_insights=portfolio_insights,
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                educational_notes=educational_notes,
                confidence_score=0.9,  # High confidence for portfolio analysis
                data_sources=["Portfolio Data", "Market Data", "Performance Metrics"],
                methodology="Comprehensive portfolio analysis including performance, risk, and diversification metrics"
            )
            
        except Exception as e:
            logger.error(f"Error generating portfolio explanation: {e}")
            return None
    
    def _generate_portfolio_insights(self,
                                   total_value: float,
                                   day_change: float,
                                   day_change_percent: float,
                                   positions: Dict[str, Any],
                                   performance: Dict[str, Any],
                                   risk_metrics: Dict[str, Any],
                                   context: ExplanationContext) -> List[Insight]:
        """Generate portfolio-specific insights"""
        
        insights = []
        
        try:
            # Performance insight
            if abs(day_change_percent) > 2:
                performance_type = "strong positive" if day_change_percent > 2 else "concerning negative"
                insights.append(Insight(
                    category="performance",
                    title=f"{performance_type.title()} Daily Performance",
                    description=f"Portfolio moved {day_change_percent:+.2f}% today (${day_change:+,.2f}), "
                               f"which is {'above' if abs(day_change_percent) > 3 else 'within'} typical daily ranges.",
                    importance=0.9,
                    confidence=1.0,
                    supporting_data={"day_change_percent": day_change_percent, "day_change": day_change}
                ))
            
            # Concentration risk insight
            if risk_metrics.get('concentration_risk', 0) > 20:
                insights.append(Insight(
                    category="risk",
                    title="High Concentration Risk Detected",
                    description=f"Your largest position represents {risk_metrics['concentration_risk']:.1f}% of your portfolio. "
                               f"Consider reducing concentration to improve diversification.",
                    importance=0.8,
                    confidence=0.9,
                    supporting_data={"concentration_risk": risk_metrics['concentration_risk']}
                ))
            
            # Overall returns insight
            if performance.get('total_return_percent'):
                return_pct = performance['total_return_percent']
                return_assessment = "excellent" if return_pct > 20 else "strong" if return_pct > 10 else "moderate" if return_pct > 0 else "negative"
                insights.append(Insight(
                    category="performance",
                    title=f"{return_assessment.title()} Overall Returns",
                    description=f"Portfolio shows {return_assessment} performance with "
                               f"{return_pct:+.1f}% total return (${performance.get('total_return', 0):+,.2f}).",
                    importance=0.8,
                    confidence=0.9,
                    supporting_data={"total_return_percent": return_pct}
                ))
            
            # Win rate insight
            if performance.get('win_rate'):
                win_rate = performance['win_rate']
                win_assessment = "high" if win_rate > 60 else "moderate" if win_rate > 40 else "low"
                insights.append(Insight(
                    category="strategy",
                    title=f"{win_assessment.title()} Success Rate",
                    description=f"Your positions show a {win_rate:.1f}% success rate "
                               f"({performance.get('winning_positions', 0)} of {performance.get('total_positions', 0)} positions positive).",
                    importance=0.7,
                    confidence=0.8,
                    supporting_data={"win_rate": win_rate}
                ))
            
            # Volatility insight
            if risk_metrics.get('volatility_score'):
                vol_score = risk_metrics['volatility_score']
                vol_level = "high" if vol_score > 15 else "low" if vol_score < 8 else "moderate"
                insights.append(Insight(
                    category="risk",
                    title=f"{vol_level.title()} Portfolio Volatility",
                    description=f"Portfolio volatility score of {vol_score:.1f} indicates {vol_level} risk profile. "
                               f"{'Consider reducing position sizes in volatile stocks' if vol_level == 'high' else 'Volatility appears well-managed' if vol_level == 'moderate' else 'Low volatility may indicate conservative positioning'}.",
                    importance=0.6,
                    confidence=0.8,
                    supporting_data={"volatility_score": vol_score}
                ))
            
            # Position count insight
            position_count = len(positions)
            if position_count < 5:
                insights.append(Insight(
                    category="diversification",
                    title="Limited Diversification",
                    description=f"With only {position_count} positions, consider adding more stocks "
                               f"across different sectors to reduce concentration risk.",
                    importance=0.7,
                    confidence=0.8,
                    supporting_data={"position_count": position_count}
                ))
            elif position_count > 20:
                insights.append(Insight(
                    category="management",
                    title="High Number of Positions",
                    description=f"Managing {position_count} positions can be challenging. "
                               f"Consider consolidating into your highest-conviction ideas.",
                    importance=0.6,
                    confidence=0.7,
                    supporting_data={"position_count": position_count}
                ))
            
            return sorted(insights, key=lambda x: x.importance, reverse=True)[:6]
            
        except Exception as e:
            logger.error(f"Error generating portfolio insights: {e}")
            return insights
    
    def _generate_portfolio_detailed_analysis(self,
                                            total_value: float,
                                            positions: Dict[str, Any],
                                            performance: Dict[str, Any],
                                            risk_metrics: Dict[str, Any],
                                            templates: Dict[str, str],
                                            context: ExplanationContext) -> str:
        """Generate detailed portfolio analysis"""
        
        try:
            analysis_parts = []
            
            # Portfolio overview
            analysis_parts.append(
                f"**Portfolio Overview:**\n"
                f"Total Value: ${total_value:,.2f}\n"
                f"Number of Positions: {len(positions)}\n"
                f"Total Return: {performance.get('total_return_percent', 0):+.1f}% "
                f"(${performance.get('total_return', 0):+,.2f})"
            )
            
            # Top positions analysis
            if positions:
                sorted_positions = sorted(positions.items(), key=lambda x: x[1].get('weight', 0), reverse=True)
                top_positions = sorted_positions[:5]
                
                analysis_parts.append("\n**Top Holdings:**")
                for symbol, pos in top_positions:
                    weight = pos.get('weight', 0)
                    pnl_percent = pos.get('day_change_percent', 0)
                    analysis_parts.append(
                        f"• {symbol}: {weight:.1f}% ({pnl_percent:+.1f}% today)"
                    )
            
            # Performance analysis
            if performance:
                win_rate = performance.get('win_rate', 0)
                analysis_parts.append(
                    f"\n**Performance Metrics:**\n"
                    f"• Success Rate: {win_rate:.1f}% of positions profitable\n"
                    f"• Winning Positions: {performance.get('winning_positions', 0)}\n"
                    f"• Total Positions: {performance.get('total_positions', 0)}"
                )
            
            # Risk analysis
            if risk_metrics:
                analysis_parts.append(
                    f"\n**Risk Assessment:**\n"
                    f"• Concentration Risk: {risk_metrics.get('concentration_risk', 0):.1f}% (largest position)\n"
                    f"• Portfolio Volatility: {risk_metrics.get('volatility_score', 0):.1f}\n"
                    f"• Diversification Score: {risk_metrics.get('diversification_score', 0):.1f}/100"
                )
            
            # Recent performance trend
            day_change = performance.get('day_change_percent', 0) if 'day_change_percent' in performance else 0
            if abs(day_change) > 1:
                trend_desc = "strong positive momentum" if day_change > 2 else "concerning decline" if day_change < -2 else "moderate movement"
                analysis_parts.append(
                    f"\n**Recent Trends:**\n"
                    f"Today's performance shows {trend_desc} with {day_change:+.2f}% movement. "
                    f"{'This outperformance may indicate good stock selection' if day_change > 1 else 'Monitor for potential reversal opportunities' if day_change < -1 else 'Performance is relatively stable'}."
                )
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating portfolio detailed analysis: {e}")
            return "Portfolio analysis data temporarily unavailable."
    
    def _generate_portfolio_recommendations(self,
                                          positions: Dict[str, Any],
                                          performance: Dict[str, Any],
                                          risk_metrics: Dict[str, Any],
                                          templates: Dict[str, str],
                                          context: ExplanationContext) -> List[str]:
        """Generate portfolio management recommendations"""
        
        recommendations = []
        
        try:
            # Concentration risk recommendations
            concentration = risk_metrics.get('concentration_risk', 0)
            if concentration > 25:
                recommendations.append(
                    f"Reduce concentration risk by trimming your largest position (currently {concentration:.1f}% of portfolio)."
                )
            elif concentration < 5 and len(positions) > 15:
                recommendations.append(
                    "Consider consolidating into your highest-conviction ideas for better focus and management."
                )
            
            # Diversification recommendations
            position_count = len(positions)
            if position_count < 8:
                recommendations.append(
                    f"Add 2-3 more positions across different sectors to improve diversification."
                )
            
            # Performance-based recommendations
            win_rate = performance.get('win_rate', 50)
            if win_rate < 40:
                recommendations.append(
                    f"Review your stock selection process - current success rate of {win_rate:.1f}% suggests room for improvement."
                )
            
            # Rebalancing recommendations
            total_return = performance.get('total_return_percent', 0)
            if total_return > 20:
                recommendations.append(
                    "Consider taking some profits and rebalancing positions that have grown significantly."
                )
            elif total_return < -10:
                recommendations.append(
                    "Review underperforming positions and consider tax-loss harvesting opportunities."
                )
            
            # Risk management recommendations
            volatility = risk_metrics.get('volatility_score', 10)
            if volatility > 20:
                recommendations.append(
                    "High portfolio volatility detected - consider reducing position sizes in the most volatile stocks."
                )
            
            # Position sizing recommendations
            if context.risk_tolerance == "conservative":
                recommendations.append(
                    "Maintain position sizes between 3-8% for individual stocks given your conservative risk profile."
                )
            elif context.risk_tolerance == "aggressive":
                recommendations.append(
                    "Consider increasing positions in your highest-conviction ideas, but maintain stop-losses."
                )
            
            # Monitoring recommendations
            recommendations.append(
                "Review portfolio allocation monthly and rebalance quarterly to maintain target weightings."
            )
            
            return recommendations[:6]  # Limit to 6 recommendations
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return ["Continue monitoring portfolio performance and maintain diversification."]
    
    def _generate_portfolio_risk_warnings(self,
                                        positions: Dict[str, Any],
                                        risk_metrics: Dict[str, Any],
                                        context: ExplanationContext) -> List[str]:
        """Generate portfolio-specific risk warnings"""
        
        warnings = []
        
        try:
            # Concentration risk warning
            concentration = risk_metrics.get('concentration_risk', 0)
            if concentration > 30:
                warnings.append(
                    f"Extreme concentration risk: {concentration:.1f}% in single position. "
                    f"Significant price movements in this stock will heavily impact your portfolio."
                )
            
            # Volatility warning
            volatility = risk_metrics.get('volatility_score', 10)
            if volatility > 25:
                warnings.append(
                    f"High portfolio volatility ({volatility:.1f}) suggests significant daily swings are likely. "
                    f"Be prepared for large portfolio value fluctuations."
                )
            
            # Diversification warning
            if len(positions) < 5:
                warnings.append(
                    f"Limited diversification with only {len(positions)} positions increases overall portfolio risk. "
                    f"A single stock's poor performance could significantly impact total returns."
                )
            
            # Performance warning
            total_return = risk_metrics.get('total_return_percent', 0)
            if total_return < -20:
                warnings.append(
                    f"Portfolio has declined significantly. Consider reviewing investment thesis "
                    f"for each position and implementing stricter risk management."
                )
            
            # General market warning
            warnings.append(
                "Market conditions can change rapidly. Maintain appropriate emergency funds "
                "and never invest more than you can afford to lose."
            )
            
            return warnings[:4]  # Limit warnings
            
        except Exception as e:
            logger.error(f"Error generating portfolio risk warnings: {e}")
            return ["Portfolio management requires ongoing attention to risk and diversification."]
    
    async def generate_market_explanation(self,
                                        context: ExplanationContext = None) -> Optional[Explanation]:
        """Generate overall market analysis and explanation"""
        
        if context is None:
            context = ExplanationContext()
        
        try:
            # Get market data
            market_service = self.market_service
            market_summary = market_service.get_market_summary()
            
            # Get market news
            market_news = await self.news_service.get_market_news()
            
            if not market_summary:
                logger.error("No market summary available")
                return None
            
            templates = self.TONE_TEMPLATES[context.preferred_tone]
            
            # Generate market insights
            market_insights = self._generate_market_insights(market_summary, market_news, context)
            
            # Create summary
            market_status = market_summary.market_status if hasattr(market_summary, 'market_status') else "unknown"
            summary = (f"{templates['intro']} the market is currently {market_status}. "
                      f"Major indices show {'mixed' if not market_news else 'positive' if market_news.market_sentiment.value.lower() == 'positive' else 'negative' if market_news.market_sentiment.value.lower() == 'negative' else 'neutral'} sentiment. ")
            
            if market_news and market_news.trending_topics:
                summary += f"Key themes include: {', '.join(market_news.trending_topics[:3])}."
            
            # Generate detailed analysis
            detailed_analysis = self._generate_market_detailed_analysis(
                market_summary, market_news, templates, context
            )
            
            # Market recommendations
            recommendations = [
                f"Monitor major indices for overall market direction and trend changes.",
                f"Stay informed about key economic indicators and Federal Reserve communications.",
                f"Maintain diversification across sectors to reduce market-specific risks."
            ]
            
            if market_news and len(market_news.breaking_news) > 3:
                recommendations.insert(0, "High volume of breaking news suggests increased volatility - consider defensive positioning.")
            
            # Market risk warnings
            risk_warnings = [
                "Market conditions can change rapidly based on economic data and geopolitical events.",
                "Individual stock performance may not follow overall market trends.",
                "Consider your risk tolerance and investment timeframe when making decisions."
            ]
            
            return Explanation(
                explanation_type=ExplanationType.MARKET_OVERVIEW,
                symbol="MARKET",
                title="Market Overview & Analysis",
                summary=summary,
                key_insights=market_insights,
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                educational_notes=self._generate_educational_notes(market_insights, context),
                confidence_score=0.8,
                data_sources=["Market Data", "News Analysis", "Sector Performance"],
                methodology="Comprehensive market analysis combining indices performance, sector rotation, and news sentiment"
            )
            
        except Exception as e:
            logger.error(f"Error generating market explanation: {e}")
            return None
    
    def _generate_market_insights(self,
                                market_summary: Any,
                                market_news: Any,
                                context: ExplanationContext) -> List[Insight]:
        """Generate market-wide insights"""
        
        insights = []
        
        try:
            # Market sentiment insight
            if market_news and hasattr(market_news, 'market_sentiment'):
                sentiment = market_news.market_sentiment.value
                insights.append(Insight(
                    category="sentiment",
                    title=f"Market Sentiment: {sentiment}",
                    description=f"Overall market sentiment appears {sentiment.lower()} "
                               f"based on news analysis across major stocks and sectors.",
                    importance=0.9,
                    confidence=0.8,
                    supporting_data={"market_sentiment": sentiment}
                ))
            
            # News volume insight
            if market_news and hasattr(market_news, 'news_volume'):
                volume = market_news.news_volume
                volume_desc = "high" if volume > 100 else "low" if volume < 30 else "moderate"
                insights.append(Insight(
                    category="activity",
                    title=f"{volume_desc.title()} News Activity",
                    description=f"Market news volume is {volume_desc} with {volume} articles analyzed. "
                               f"{'This suggests heightened market attention' if volume_desc == 'high' else 'Limited news flow may indicate stable conditions' if volume_desc == 'low' else 'Normal level of market coverage'}.",
                    importance=0.7,
                    confidence=0.8,
                    supporting_data={"news_volume": volume}
                ))
            
            # Breaking news insight
            if market_news and hasattr(market_news, 'breaking_news') and len(market_news.breaking_news) > 0:
                breaking_count = len(market_news.breaking_news)
                insights.append(Insight(
                    category="news",
                    title="Breaking News Impact",
                    description=f"{breaking_count} breaking news items detected across the market. "
                               f"Expect increased volatility and monitor for sector-specific impacts.",
                    importance=0.9,
                    confidence=0.9,
                    supporting_data={"breaking_news_count": breaking_count}
                ))
            
            # Trending topics insight
            if market_news and hasattr(market_news, 'trending_topics') and market_news.trending_topics:
                top_topics = market_news.trending_topics[:5]
                insights.append(Insight(
                    category="themes",
                    title="Market Themes",
                    description=f"Key topics driving market discussion: {', '.join(top_topics)}. "
                               f"These themes may influence sector rotation and individual stock performance.",
                    importance=0.8,
                    confidence=0.7,
                    supporting_data={"trending_topics": top_topics}
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return insights
    
    def _generate_market_detailed_analysis(self,
                                         market_summary: Any,
                                         market_news: Any,
                                         templates: Dict[str, str],
                                         context: ExplanationContext) -> str:
        """Generate detailed market analysis"""
        
        try:
            analysis_parts = []
            
            # Market status
            status = getattr(market_summary, 'market_status', 'unknown')
            analysis_parts.append(
                f"**Market Status:** {status.title()}"
            )
            
            # Sentiment analysis
            if market_news:
                sentiment = getattr(market_news, 'market_sentiment', None)
                if sentiment:
                    analysis_parts.append(
                        f"**Market Sentiment:** {sentiment.value} based on comprehensive news analysis"
                    )
                
                # Trending topics
                topics = getattr(market_news, 'trending_topics', [])
                if topics:
                    analysis_parts.append(
                        f"**Key Themes:** {', '.join(topics[:5])}"
                    )
                
                # Breaking news
                breaking = getattr(market_news, 'breaking_news', [])
                if breaking:
                    analysis_parts.append(
                        f"**Breaking News:** {len(breaking)} urgent developments detected - monitor for volatility"
                    )
            
            # Add sector analysis if available
            if market_news and hasattr(market_news, 'sector_sentiment'):
                sector_data = market_news.sector_sentiment
                if sector_data:
                    analysis_parts.append("**Sector Performance:**")
                    for sector, sentiment in sector_data.items():
                        if sentiment != 0:
                            analysis_parts.append(f"• {sector}: {sentiment:+.2f}")
            
            # Market implications
            analysis_parts.append(
                f"\n**Market Implications:**\n"
                f"Current conditions suggest {'increased caution' if status == 'closed' else 'active monitoring'} "
                f"for individual stock selection and portfolio management. "
                f"{'Focus on defensive positioning' if market_news and hasattr(market_news, 'market_sentiment') and 'negative' in market_news.market_sentiment.value.lower() else 'Look for selective opportunities' if market_news else 'Maintain balanced approach'}."
            )
            
            return "\n\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating market detailed analysis: {e}")
            return "Market analysis data temporarily unavailable."
    
    def health_check(self) -> Dict[str, Any]:
        """Perform explanation service health check"""
        
        try:
            start_time = time.time()
            
            # Test basic explanation generation
            test_context = ExplanationContext()
            test_insight = Insight(
                category="test",
                title="Test Insight",
                description="This is a test insight",
                importance=0.5,
                confidence=0.8
            )
            
            # Test template generation
            templates = self.TONE_TEMPLATES[ToneStyle.CONVERSATIONAL]
            template_test = bool(templates and "intro" in templates)
            
            # Test service integrations
            market_healthy = hasattr(self.market_service, 'get_stock_quote')
            prediction_healthy = hasattr(self.prediction_service, 'predict_stock_price')
            news_healthy = hasattr(self.news_service, 'analyze_sentiment')
            cache_healthy = self.cache.get_stats().get('status') == 'connected'
            
            response_time = time.time() - start_time
            
            components = {
                "template_system": "healthy" if template_test else "unhealthy",
                "market_service": "healthy" if market_healthy else "unhealthy", 
                "prediction_service": "healthy" if prediction_healthy else "unhealthy",
                "news_service": "healthy" if news_healthy else "unhealthy",
                "cache_service": "healthy" if cache_healthy else "unhealthy"
            }
            
            overall_healthy = all(status == "healthy" for status in components.values())
            
            return {
                "service": "explanation_service",
                "status": "healthy" if overall_healthy else "unhealthy",
                "components": components,
                "features": {
                    "explanation_types": len(ExplanationType),
                    "complexity_levels": len(ComplexityLevel),
                    "tone_styles": len(ToneStyle),
                    "educational_terms": len(self.EDUCATIONAL_TERMS)
                },
                "response_time": round(response_time, 3),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "service": "explanation_service",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global service instance
_explanation_service = None

def get_explanation_service() -> ExplanationService:
    """Get global explanation service instance"""
    global _explanation_service
    if _explanation_service is None:
        _explanation_service = ExplanationService()
    return _explanation_service

# Convenience functions for different types of explanations
async def explain_stock(symbol: str, complexity: str = "intermediate", tone: str = "conversational") -> Optional[Explanation]:
    """Convenience function for stock analysis explanation"""
    service = get_explanation_service()
    context = ExplanationContext(
        user_experience_level=ComplexityLevel(complexity),
        preferred_tone=ToneStyle(tone)
    )
    return await service.explain_stock_analysis(symbol, context)

async def explain_signal(symbol: str, signal: TradingSignal, complexity: str = "intermediate") -> Optional[Explanation]:
    """Convenience function for trading signal explanation"""
    service = get_explanation_service()
    context = ExplanationContext(
        user_experience_level=ComplexityLevel(complexity)
    )
    return await service.explain_trading_signal(symbol, signal, context)

async def explain_portfolio(user_id: str, portfolio_data: Dict, risk_tolerance: str = "moderate") -> Optional[Explanation]:
    """Convenience function for portfolio explanation"""
    service = get_explanation_service()
    context = ExplanationContext(
        risk_tolerance=risk_tolerance
    )
    return await service.generate_portfolio_explanation(user_id, portfolio_data, context)

async def explain_price_move(symbol: str, timeframe: str = "1d") -> Optional[Explanation]:
    """Convenience function for price movement explanation"""
    service = get_explanation_service()
    return await service.explain_price_movement(symbol, timeframe)

# Educational content generator
class EducationalContentGenerator:
    """Generate educational content for financial concepts"""
    
    @staticmethod
    def explain_concept(concept: str, level: ComplexityLevel = ComplexityLevel.INTERMEDIATE) -> str:
        """Explain financial concepts at different complexity levels"""
        
        explanations = {
            "technical_analysis": {
                ComplexityLevel.BEGINNER: "Technical analysis studies price charts and patterns to predict future price movements. It's like reading the stock's 'mood' through its price history.",
                ComplexityLevel.INTERMEDIATE: "Technical analysis uses historical price data, volume, and mathematical indicators to identify trends and potential entry/exit points. Common tools include moving averages, RSI, and MACD.",
                ComplexityLevel.ADVANCED: "Technical analysis employs quantitative methods to analyze price action, volume patterns, and momentum indicators. It assumes that all relevant information is reflected in price and that prices move in trends that tend to persist."
            },
            "fundamental_analysis": {
                ComplexityLevel.BEGINNER: "Fundamental analysis looks at a company's financial health, earnings, and growth prospects to determine if a stock is fairly priced.",
                ComplexityLevel.INTERMEDIATE: "Fundamental analysis evaluates a company's intrinsic value by examining financial statements, industry position, management quality, and economic factors. Key metrics include P/E ratio, revenue growth, and debt levels.",
                ComplexityLevel.ADVANCED: "Fundamental analysis involves comprehensive evaluation of quantitative factors (financial ratios, cash flow analysis, earnings quality) and qualitative factors (competitive advantages, management effectiveness, regulatory environment) to determine intrinsic value."
            },
            "diversification": {
                ComplexityLevel.BEGINNER: "Diversification means not putting all your eggs in one basket. Spread your investments across different stocks and sectors to reduce risk.",
                ComplexityLevel.INTERMEDIATE: "Diversification reduces portfolio risk by investing across different asset classes, sectors, and geographic regions. The goal is to minimize the impact of any single investment's poor performance on the overall portfolio.",
                ComplexityLevel.ADVANCED: "Diversification optimizes the risk-return profile through strategic asset allocation based on correlation coefficients, systematic vs. unsystematic risk, and efficient frontier theory. Modern Portfolio Theory suggests optimal diversification can reduce risk without sacrificing expected return."
            },
            "risk_management": {
                ComplexityLevel.BEGINNER: "Risk management is about protecting your money. Never invest more than you can afford to lose, and always have a plan for when things go wrong.",
                ComplexityLevel.INTERMEDIATE: "Risk management involves position sizing, stop-loss orders, diversification, and regular portfolio review. The goal is to limit downside while allowing for upside potential. Common techniques include the 1-2% rule for position sizing.",
                ComplexityLevel.ADVANCED: "Risk management encompasses systematic approaches including Value at Risk (VaR) calculations, beta-adjusted position sizing, options hedging strategies, and dynamic risk budgeting. It requires continuous monitoring of portfolio risk metrics and stress testing under various market scenarios."
            }
        }
        
        if concept in explanations and level in explanations[concept]:
            return explanations[concept][level]
        else:
            return f"Educational content for '{concept}' at {level.value} level is not yet available."

# Chat integration helpers
class ChatResponseGenerator:
    """Generate conversational responses for AI chat interface"""
    
    @staticmethod
    def generate_stock_chat_response(explanation: Explanation, user_question: str = "") -> str:
        """Convert explanation to conversational chat response"""
        
        if not explanation:
            return "I'm sorry, I couldn't analyze that stock right now. Please try again or ask about a different symbol."
        
        # Start with a conversational greeting
        chat_response = f"I've analyzed {explanation.symbol} for you! "
        
        # Add summary in conversational tone
        chat_response += explanation.summary + "\n\n"
        
        # Add key insights in bullet format
        if explanation.key_insights:
            chat_response += "Here are the key things I found:\n"
            for insight in explanation.key_insights[:3]:  # Top 3 insights
                chat_response += f"• {insight.description}\n"
            chat_response += "\n"
        
        # Add top recommendations
        if explanation.recommendations:
            chat_response += "My recommendations:\n"
            for rec in explanation.recommendations[:2]:  # Top 2 recommendations
                chat_response += f"• {rec}\n"
            chat_response += "\n"
        
        # Add important warnings
        if explanation.risk_warnings:
            chat_response += f"⚠️ Important: {explanation.risk_warnings[0]}\n\n"
        
        # Confidence and source info
        chat_response += f"Confidence level: {explanation.confidence_score:.0%} | Sources: {', '.join(explanation.data_sources[:3])}"
        
        return chat_response
    
    @staticmethod
    def generate_simple_response(message: str, tone: ToneStyle = ToneStyle.CONVERSATIONAL) -> str:
        """Generate simple AI responses for basic questions"""
        
        templates = {
            ToneStyle.CONVERSATIONAL: {
                "greeting": "Hi! I'm your AI financial advisor. I can help you analyze stocks, explain market movements, and provide investment insights. What would you like to know?",
                "help": "I can help you with:\n• Stock analysis and predictions\n• Trading signal explanations\n• Portfolio reviews\n• Market overviews\n• Educational content about investing\n\nJust ask me about any stock symbol or investment topic!",
                "error": "I'm having trouble with that request right now. Could you try rephrasing your question or ask about something else?",
                "unknown": "I'm not sure about that specific question, but I'd be happy to help analyze a stock, explain a trading concept, or review market conditions. What interests you most?"
            },
            ToneStyle.PROFESSIONAL: {
                "greeting": "Welcome to your AI Financial Advisory Service. I provide comprehensive investment analysis, market insights, and portfolio guidance. How may I assist you today?",
                "help": "Our services include:\n• Comprehensive stock analysis\n• Technical and fundamental insights\n• Risk assessment and portfolio optimization\n• Market trend analysis\n• Investment education\n\nPlease specify your area of interest for detailed analysis.",
                "error": "We are currently experiencing technical difficulties. Please retry your request or contact support for assistance.",
                "unknown": "That inquiry falls outside my current analytical capabilities. I recommend focusing on equity analysis, market research, or portfolio management topics."
            }
        }
        
        message_lower = message.lower()
        template_set = templates.get(tone, templates[ToneStyle.CONVERSATIONAL])
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "start"]):
            return template_set["greeting"]
        elif any(word in message_lower for word in ["help", "what can you", "how do you"]):
            return template_set["help"]
        elif any(word in message_lower for word in ["error", "problem", "issue"]):
            return template_set["error"]
        else:
            return template_set["unknown"]

# Market alerts and notifications
class AlertGenerator:
    """Generate market alerts and notifications"""
    
    @staticmethod
    def generate_price_alert(symbol: str, current_price: float, target_price: float, alert_type: str) -> Dict[str, Any]:
        """Generate price-based alerts"""
        
        price_change = ((current_price - target_price) / target_price) * 100
        
        return {
            "type": "price_alert",
            "symbol": symbol,
            "current_price": current_price,
            "target_price": target_price,
            "price_change_percent": price_change,
            "alert_type": alert_type,
            "message": f"{symbol} has {'reached' if alert_type == 'target' else 'broken'} ${target_price:.2f} (currently ${current_price:.2f})",
            "urgency": "high" if abs(price_change) > 5 else "medium" if abs(price_change) > 2 else "low",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def generate_news_alert(symbol: str, news_impact: str, sentiment_change: float) -> Dict[str, Any]:
        """Generate news-based alerts"""
        
        return {
            "type": "news_alert",
            "symbol": symbol,
            "impact": news_impact,
            "sentiment_change": sentiment_change,
            "message": f"Breaking news may impact {symbol} - {news_impact} with sentiment shift of {sentiment_change:+.2f}",
            "urgency": "high" if news_impact == "major" else "medium",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Integration helpers for frontend
class FrontendIntegration:
    """Helper functions for frontend integration"""
    
    @staticmethod
    def format_for_react(explanation: Explanation) -> Dict[str, Any]:
        """Format explanation data for React components"""
        
        return {
            "type": explanation.explanation_type.value,
            "symbol": explanation.symbol,
            "title": explanation.title,
            "summary": explanation.summary,
            "keyInsights": [
                {
                    "category": insight.category,
                    "title": insight.title,
                    "description": insight.description,
                    "importance": insight.importance,
                    "confidence": insight.confidence,
                    "supportingData": insight.supporting_data
                }
                for insight in explanation.key_insights
            ],
            "detailedAnalysis": explanation.detailed_analysis,
            "recommendations": explanation.recommendations,
            "riskWarnings": explanation.risk_warnings,
            "educationalNotes": explanation.educational_notes,
            "confidenceScore": explanation.confidence_score,
            "dataSources": explanation.data_sources,
            "methodology": explanation.methodology,
            "timestamp": explanation.timestamp.isoformat()
        }
    
    @staticmethod
    def generate_dashboard_summary(explanations: List[Explanation]) -> Dict[str, Any]:
        """Generate summary for dashboard display"""
        
        if not explanations:
            return {"error": "No explanations available"}
        
        # Count explanation types
        type_counts = {}
        for exp in explanations:
            exp_type = exp.explanation_type.value
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        # Average confidence
        avg_confidence = sum(exp.confidence_score for exp in explanations) / len(explanations)
        
        # Recent insights
        all_insights = []
        for exp in explanations:
            all_insights.extend(exp.key_insights[:2])  # Top 2 from each
        
        # Sort by importance and recency
        recent_insights = sorted(all_insights, key=lambda x: (x.importance, x.timestamp if hasattr(x, 'timestamp') else datetime.now()), reverse=True)[:5]
        
        return {
            "totalExplanations": len(explanations),
            "typeBreakdown": type_counts,
            "averageConfidence": round(avg_confidence, 2),
            "recentInsights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "category": insight.category,
                    "importance": insight.importance
                }
                for insight in recent_insights
            ],
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_explanation_service():
        """Test the explanation service functionality"""
        
        print("🧠 Testing AI Explanation Service...\n")
        
        service = ExplanationService()
        
        # Health check
        health = service.health_check()
        print(f"🏥 Health Check: {health['status'].upper()}")
        print(f"📊 Components: {health['components']}")
        
        print("\n" + "="*60 + "\n")
        
        # Test stock analysis explanation
        try:
            print("📈 Testing Stock Analysis Explanation...")
            
            context = ExplanationContext(
                user_experience_level=ComplexityLevel.INTERMEDIATE,
                preferred_tone=ToneStyle.CONVERSATIONAL,
                include_educational=True
            )
            
            explanation = await service.explain_stock_analysis("AAPL", context)
            
            if explanation:
                print(f"✅ Generated explanation: {explanation.title}")
                print(f"📝 Summary: {explanation.summary[:150]}...")
                print(f"🔍 Key Insights: {len(explanation.key_insights)}")
                print(f"💡 Recommendations: {len(explanation.recommendations)}")
                print(f"⚠️  Risk Warnings: {len(explanation.risk_warnings)}")
                print(f"🎯 Confidence: {explanation.confidence_score:.0%}")
                
                # Test chat response generation
                chat_response = ChatResponseGenerator.generate_stock_chat_response(explanation)
                print(f"💬 Chat Response: {len(chat_response)} characters")
                
                # Test React formatting
                react_data = FrontendIntegration.format_for_react(explanation)
                print(f"⚛️  React Format: {len(react_data)} fields")
            else:
                print("❌ No explanation generated")
                
        except Exception as e:
            print(f"❌ Stock analysis test failed: {e}")
        
        print("\n" + "-"*40 + "\n")
        
        # Test educational content
        print("📚 Testing Educational Content...")
        try:
            concepts = ["technical_analysis", "diversification", "risk_management"]
            levels = [ComplexityLevel.BEGINNER, ComplexityLevel.INTERMEDIATE, ComplexityLevel.ADVANCED]
            
            for concept in concepts:
                for level in levels:
                    content = EducationalContentGenerator.explain_concept(concept, level)
                    print(f"• {concept} ({level.value}): {len(content)} chars")
                    
        except Exception as e:
            print(f"❌ Educational content test failed: {e}")
        
        print("\n" + "-"*40 + "\n")
        
        # Test price movement explanation
        print("📊 Testing Price Movement Explanation...")
        try:
            movement_explanation = await service.explain_price_movement("AAPL", "1d")
            if movement_explanation:
                print(f"✅ Movement explanation: {movement_explanation.title}")
                print(f"📈 Insights: {len(movement_explanation.key_insights)}")
            else:
                print("ℹ️  No significant price movement to explain")
                
        except Exception as e:
            print(f"❌ Price movement test failed: {e}")
        
        print("\n" + "-"*40 + "\n")
        
        # Test alert generation
        print("🚨 Testing Alert Generation...")
        try:
            price_alert = AlertGenerator.generate_price_alert("AAPL", 150.0, 145.0, "support")
            news_alert = AlertGenerator.generate_news_alert("AAPL", "major", -0.3)
            
            print(f"📈 Price Alert: {price_alert['message']}")
            print(f"📰 News Alert: {news_alert['message']}")
            
        except Exception as e:
            print(f"❌ Alert generation test failed: {e}")
        
        print("\n✅ Explanation Service Test Completed!")
    
    
    
    def _generate_news_section(self,
                             symbol: str,
                             sentiment: SentimentAnalysis,
                             templates: Dict[str, str],
                             context: ExplanationContext) -> str:
        """Generate news and sentiment section"""
        
        news_parts = [f"**News & Sentiment Analysis:**"]
        
        # Overall sentiment
        sentiment_desc = sentiment.overall_sentiment.value.lower()
        news_parts.append(
            f"Recent news sentiment for {symbol} appears {sentiment_desc} "
            f"based on analysis of {sentiment.total_articles} articles "
            f"(sentiment score: {sentiment.sentiment_score:+.2f})."
        )
        
        # Article distribution
        news_parts.append(
            f"Article breakdown: {sentiment.positive_articles} positive, "
            f"{sentiment.negative_articles} negative, {sentiment.neutral_articles} neutral."
        )
        
        # Trending topics
        if sentiment.trending_keywords:
            news_parts.append(
                f"Key topics in the news: {', '.join(sentiment.trending_keywords[:5])}."
            )
        
        # Sentiment trend
        if sentiment.sentiment_trend != "stable":
            news_parts.append(f"Sentiment appears to be {sentiment.sentiment_trend} over time.")
        
        # Breaking news impact
        if sentiment.recent_breaking_news:
            news_parts.append(
                f"{templates['caution']} {len(sentiment.recent_breaking_news)} breaking news articles "
                f"may cause increased volatility."
            )
        
        # News volume assessment
        volume_impact = {
            "high": "Heavy news coverage suggests heightened investor attention.",
            "normal": "Moderate news flow indicates typical market interest.",
            "low": "Limited news coverage may mean less immediate catalysts."
        }
        if sentiment.news_volume in volume_impact:
            news_parts.append(volume_impact[sentiment.news_volume])
        
        return " ".join(news_parts)
    
    def _generate_insights_section(self,
                                 insights: List[Insight],
                                 templates: Dict[str, str],
                                 context: ExplanationContext) -> str:
        """Generate key insights section"""
        
        if not insights:
            return ""
        
        insights_parts = [f"**Key Insights:**"]
        
        # Group insights by category
        categories = {}
        for insight in insights[:6]:  # Top 6 insights
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight)
        
        for category, cat_insights in categories.items():
            category_title = category.replace("_", " ").title()
            insights_parts.append(f"*{category_title}:*")
            
            for insight in cat_insights:
                confidence_text = f" (Confidence: {insight.confidence:.0%})" if context.user_experience_level == ComplexityLevel.ADVANCED else ""
                insights_parts.append(f"• {insight.description}{confidence_text}")
        
        return " ".join(insights_parts)
    
    def _generate_recommendations(self,
                                symbol: str,
                                quote: StockQuote,
                                prediction: Optional[PredictionResult],
                                signal: Optional[TradingSignal],
                                sentiment: Optional[SentimentAnalysis],
                                templates: Dict[str, str],
                                context: ExplanationContext) -> List[str]:
        """Generate actionable recommendations"""
        
        try:
            recommendations = []
            
            # Signal-based recommendations
            if signal:
                if signal.signal == SignalType.STRONG_BUY:
                    recommendations.append(
                        f"{templates['recommendation']} considering a strong buy position in {symbol} "
                        f"with a target price of ${signal.target_price:.2f}."
                    )
                    if signal.stop_loss:
                        recommendations.append(f"Set a stop-loss at ${signal.stop_loss:.2f} to manage risk.")
                
                elif signal.signal == SignalType.BUY:
                    recommendations.append(
                        f"{templates['recommendation']} gradually building a position in {symbol}."
                    )
                
                elif signal.signal == SignalType.STRONG_SELL:
                    recommendations.append(
                        f"{templates['recommendation']} considering reducing or exiting your {symbol} position."
                    )
                
                elif signal.signal == SignalType.SELL:
                    recommendations.append(
                        f"{templates['recommendation']} monitoring {symbol} closely for potential exit points."
                    )
                
                else:  # HOLD
                    recommendations.append(
                        f"{templates['recommendation']} maintaining your current {symbol} position and monitoring developments."
                    )
            
            # Risk-based recommendations
            if prediction and prediction.volatility_score > 0.3:
                recommendations.append(
                    "Consider smaller position sizes due to higher volatility levels."
                )
            
            # Sentiment-based recommendations
            if sentiment and sentiment.recent_breaking_news:
                recommendations.append(
                    "Monitor breaking news developments that may impact price action."
                )
            
            # Time horizon recommendations
            if context.investment_timeframe == "short_term":
                recommendations.append(
                    "For short-term trading, pay close attention to technical levels and volume."
                )
            elif context.investment_timeframe == "long_term":
                recommendations.append(
                    "For long-term investing, focus on fundamental strength and industry trends."
                )
            
            # Risk tolerance adjustments
            if context.risk_tolerance == "conservative":
                recommendations.append(
                    "Given your conservative approach, consider waiting for clearer signals or dollar-cost averaging."
                )
            elif context.risk_tolerance == "aggressive":
                recommendations.append(
                    "With higher risk tolerance, you might consider larger position sizes on strong signals."
                )
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append(
                    f"Continue monitoring {symbol} and maintain appropriate position sizing for your risk profile."
                )
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Monitor {symbol} closely and consult additional sources before making decisions."]
    
    def _generate_risk_warnings(self,
                              symbol: str,
                              quote: StockQuote,
                              prediction: Optional[PredictionResult],
                              signal: Optional[TradingSignal],
                              sentiment: Optional[SentimentAnalysis],
                              context: ExplanationContext) -> List[str]:
        """Generate risk warnings"""
        
        warnings = []
        
        try:
            # High volatility warning
            if prediction and prediction.volatility_score > 0.3:
                warnings.append(
                    f"High volatility detected ({prediction.volatility_score:.2f}). "
                    "Price swings may be significant."
                )
            
            # Large price movement warning
            if abs(quote.change_percent) > 5:
                warnings.append(
                    f"Unusual price movement of {abs(quote.change_percent):.1f}% may indicate "
                    "increased risk or opportunity."
                )
            
            # News impact warning
            if sentiment and sentiment.recent_breaking_news:
                warnings.append(
                    "Breaking news may cause unpredictable price movements."
                )
            
            # Low confidence warning
            confidence_sources = []
            if prediction and prediction.confidence_score < 0.6:
                confidence_sources.append("prediction model")
            if signal and signal.confidence < 0.6:
                confidence_sources.append("trading signals")
            if sentiment and sentiment.confidence < 0.6:
                confidence_sources.append("sentiment analysis")
            
            if confidence_sources:
                warnings.append(
                    f"Lower confidence in {', '.join(confidence_sources)}. "
                    "Exercise additional caution."
                )
            
            # Standard disclaimer
            warnings.append(
                "All investments carry risk. Past performance does not guarantee future results."
            )
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error generating risk warnings: {e}")
            return ["All investments carry risk. Please do your own research."]
    
    def _generate_educational_notes(self,
                                  insights: List[Insight],
                                  context: ExplanationContext) -> List[str]:
        """Generate educational explanations"""
        
        if not context.include_educational:
            return []
        
        educational_notes = []
        
        try:
            # Extract technical terms mentioned in insights
            terms_mentioned = set()
            for insight in insights:
                description_lower = insight.description.lower()
                for term in self.EDUCATIONAL_TERMS.keys():
                    if term.lower() in description_lower:
                        terms_mentioned.add(term)
            
            # Add explanations for mentioned terms
            for term in sorted(terms_mentioned):
                if term in self.EDUCATIONAL_TERMS:
                    educational_notes.append(f"**{term}**: {self.EDUCATIONAL_TERMS[term]}")
            
            # Add general educational notes based on context
            if context.user_experience_level == ComplexityLevel.BEGINNER:
                educational_notes.extend([
                    "**Diversification**: Don't put all your money in one stock. Spread risk across different investments.",
                    "**Dollar-Cost Averaging**: Consider investing fixed amounts regularly rather than all at once.",
                    "**Research**: Always do additional research before making investment decisions."
                ])
            
            return educational_notes[:6]  # Limit to 6 notes
            
        except Exception as e:
            logger.error(f"Error generating educational notes: {e}")
            return []
    
    def _calculate_explanation_confidence(self,
                                        quote: StockQuote,
                                        prediction: Optional[PredictionResult],
                                        signal: Optional[TradingSignal],
                                        sentiment: Optional[SentimentAnalysis],
                                        insights: List[Insight]) -> float:
        """Calculate overall confidence score for the explanation"""
        
        try:
            confidence_scores = []
            
            # Data availability confidence
            data_confidence = 0.5  # Base confidence for having quote data
            
            if prediction:
                data_confidence += 0.2
                confidence_scores.append(prediction.confidence_score)
            
            if signal:
                data_confidence += 0.2
                confidence_scores.append(signal.confidence)
            
            if sentiment and sentiment.total_articles >= 3:
                data_confidence += 0.1
                confidence_scores.append(sentiment.confidence)
            
            confidence_scores.append(data_confidence)
            
            # Insight quality confidence
            if insights:
                avg_insight_confidence = sum(insight.confidence for insight in insights) / len(insights)
                confidence_scores.append(avg_insight_confidence)
            
            # Calculate weighted average
            if confidence_scores:
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
                return min(max(overall_confidence, 0.0), 1.0)  # Clamp between 0 and 1
            
            return 0.5  # Default moderate confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def explain_trading_signal(self,
                                   signal: TradingSignal,
                                   context: ExplanationContext = None) -> Optional[Explanation]:
        """Generate explanation for a trading signal"""
        
        if context is None:
            context = ExplanationContext()
        
        try:
            templates = self.TONE_TEMPLATES[context.preferred_tone]
            
            # Create insights for the signal
            insights = [
                Insight(
                    category="technical",
                    title=f"{signal.signal.value.replace('_', ' ').title()} Signal",
                    description=f"Generated {signal.signal.value.replace('_', ' ').lower()} signal "
                               f"with {signal.confidence:.0%} confidence. "
                               f"Key factors: {', '.join(signal.reasoning[:3])}.",
                    importance=0.9,
                    confidence=signal.confidence,
                    supporting_data={
                        "signal_type": signal.signal.value,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                        "reasoning": signal.reasoning
                    }
                )
            ]
            
            return Explanation(
                explanation_type=ExplanationType.TRADING_SIGNAL,
                symbol=signal.symbol,
                title=f"{signal.symbol} Trading Signal Analysis",
                summary=f"{templates['intro']} {signal.symbol} shows a {signal.signal.value.replace('_', ' ').lower()} signal.",
                key_insights=insights,
                detailed_analysis=f"Technical analysis indicates {signal.signal.value.replace('_', ' ').lower()} "
                                f"conditions for {signal.symbol}. " + " ".join(signal.reasoning),
                recommendations=[f"{templates['recommendation']} acting on this {signal.signal.value.replace('_', ' ').lower()} signal."],
                risk_warnings=["Trading signals are not guarantees. Use proper risk management."],
                educational_notes=self._generate_educational_notes(insights, context),
                confidence_score=signal.confidence,
                data_sources=["Technical Analysis"],
                methodology="Multi-indicator technical analysis with machine learning validation"
            )
            
        except Exception as e:
            logger.error(f"Error explaining trading signal: {e}")
            return None


# Global service instance
_explanation_service = None

def get_explanation_service(
    cache_service: CacheService = None,
    market_service: MarketDataService = None,
    prediction_service: PredictionService = None,
    news_service: NewsService = None
) -> ExplanationService:
    """Get or create explanation service instance"""
    global _explanation_service
    
    if _explanation_service is None:
        _explanation_service = ExplanationService(
            cache_service=cache_service,
            market_service=market_service,
            prediction_service=prediction_service,
            news_service=news_service
        )
        logger.info("Created new explanation service instance")
    
    return _explanation_service

def reset_explanation_service():
    """Reset explanation service instance (useful for testing)"""
    global _explanation_service
    _explanation_service = None
    logger.info("Reset explanation service instance")


# Test and example usage
if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_explanation_service():
        print("🚀 Testing Explanation Service...\n")
        
        # Initialize service
        service = get_explanation_service()
        
        # Test stock analysis
        try:
            context = ExplanationContext(
                user_experience_level=ComplexityLevel.INTERMEDIATE,
                preferred_tone=ToneStyle.CONVERSATIONAL,
                risk_tolerance="moderate",
                investment_timeframe="medium_term"
            )
            
            print("📊 Generating stock analysis explanation...")
            explanation = await service.explain_stock_analysis("AAPL", context)
            
            if explanation:
                print(f"✅ Generated explanation for {explanation.symbol}")
                print(f"📈 Title: {explanation.title}")
                print(f"📝 Summary: {explanation.summary}")
                print(f"🎯 Confidence: {explanation.confidence_score:.0%}")
                print(f"💡 Key insights: {len(explanation.key_insights)}")
                print(f"🔍 Recommendations: {len(explanation.recommendations)}")
            else:
                print("❌ Failed to generate explanation")
                
        except Exception as e:
            print(f"❌ Error during testing: {e}")
    
    async def main():
        await test_explanation_service()
    
    # Run the async test
    asyncio.run(main())