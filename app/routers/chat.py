"""
Enhanced chat.py Router
Transforms basic chat into sophisticated AI Financial Advisor
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import re
from pydantic import BaseModel

# Your existing imports (kept for compatibility)
from app.db.db import get_db
from app.schemas.chat import ChatRequest, ChatResponse

# New service imports (your enhanced backend)
from app.services.explanation_services import get_explanation_service, ExplanationService, ExplanationContext, ComplexityLevel, ToneStyle, ChatResponseGenerator

from app.services.data_collector import get_data_collector, DataCollector
from app.services.market_data_services import get_market_service, MarketDataService
from app.services.prediction_services import get_prediction_service, PredictionService
from app.services.news_services import get_news_service, NewsService
from app.services.cache import get_cache_service, CacheType
from app.services.llm_service import get_llm_service, LLMService, Message, ConversationRole
# Legacy imports (for compatibility)
from app.services.market_data_services import fetch_historical_data, to_df
from app.services.prediction_services import add_indicators


# FIX: Ensure llm_services.py exists in app/services or update the import path accordingly.


router = APIRouter(prefix="/api/chat", tags=["AI Chat"])

# Enhanced Pydantic models
class EnhancedChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    complexity_level: str = "intermediate"
    tone: str = "conversational"
    conversation_history: Optional[List[Dict[str, str]]] = None
    use_llm: bool = True  # 🔥 ADD THIS LINE

class EnhancedChatResponse(BaseModel):
    message: str
    ticker: Optional[str] = None
    response_type: str  # analysis, explanation, general, error
    analytics: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    confidence_score: float
    data_sources: List[str]
    timestamp: str
    conversation_id: Optional[str] = None
    powered_by: str = "AI"  # 🔥 NEW: Indicates if LLM was used

class ConversationHistory(BaseModel):
    user_id: str
    conversation_id: str
    messages: List[Dict[str, Any]]
    created_at: str
    updated_at: str

class ChatAnalyticsRequest(BaseModel):
    ticker: str
    analysis_type: str = "comprehensive"  # basic, comprehensive, signals, explanation
    user_context: Optional[Dict[str, Any]] = None

# Dependency injection
def get_explanation_service_dep() -> ExplanationService:
    return get_explanation_service()

def get_data_collector_dep() -> DataCollector:
    return get_data_collector()

def get_market_data_service_dep() -> MarketDataService:
    return get_market_service()

def get_prediction_service_dep() -> PredictionService:
    return get_prediction_service()

def get_news_service_dep() -> NewsService:
    return get_news_service()

# Conversation state management
active_conversations = {}

class ChatService:
    """Enhanced chat service with AI capabilities"""
    
    def __init__(self):
        self.explanation_service = get_explanation_service()
        self.data_collector = get_data_collector()
        self.market_service = get_market_service()
        self.prediction_service = get_prediction_service()
        self.news_service = get_news_service()
        self.cache_service = get_cache_service()

        try:
            self.llm_service = get_llm_service()
            self.llm_enabled = True
        except Exception as e:
            print(f"⚠️ LLM service not available: {e}")
            self.llm_service = None
            self.llm_enabled = False
    
    def extract_ticker_from_message(self, message: str) -> Optional[str]:
        """Extract stock ticker from user message"""
        # Common patterns for ticker extraction
        patterns = [
            r'\b([A-Z]{1,5})\b(?:\s+stock|\s+shares|\s+price)?',  # AAPL stock
            r'(?:stock|ticker|symbol)\s+([A-Z]{1,5})\b',  # stock AAPL
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'(?:analyze|analysis)\s+([A-Z]{1,5})\b',  # analyze AAPL
            r'(?:buy|sell|hold)\s+([A-Z]{1,5})\b',  # buy AAPL
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.upper())
            if match:
                ticker = match.group(1)
                # Validate ticker length (1-5 chars)
                if 1 <= len(ticker) <= 5:
                    return ticker
        return None
    
    def determine_intent(self, message: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Determine user intent from message"""
        message_lower = message.lower()
        
        intents = {
            'stock_analysis': ['analyze', 'analysis', 'report', 'overview', 'summary'],
            'buy_sell_advice': ['buy', 'sell', 'hold', 'invest', 'should i', 'recommend'],
            'price_prediction': ['predict', 'forecast', 'target', 'price', 'will go', 'future'],
            'news_sentiment': ['news', 'sentiment', 'articles', 'headlines', 'breaking'],
            'technical_analysis': ['rsi', 'macd', 'technical', 'indicators', 'chart', 'support', 'resistance'],
            'portfolio_help': ['portfolio', 'diversify', 'allocation', 'risk', 'balance'],
            'educational': ['explain', 'what is', 'how does', 'meaning', 'definition', 'teach me'],
            'market_overview': ['market', 'indices', 'economy', 'sector', 'overall'],
            'greeting': ['hello', 'hi', 'hey', 'start', 'help'],
            'comparison': ['compare', 'vs', 'versus', 'better', 'difference'],
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Determine primary intent
        primary_intent = detected_intents[0] if detected_intents else 'general'
        
        return {
            'primary_intent': primary_intent,
            'all_intents': detected_intents,
            'has_ticker': ticker is not None,
            'ticker': ticker,
            'confidence': 0.8 if detected_intents else 0.3
        }
    
    
    async def _gather_context_for_llm(
    self, 
    ticker: Optional[str],
    intent_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Gather context data for LLM"""
        if not ticker:
            return None
    
        context = {}
    
        try:
             # Get stock quote
            quote = self.market_service.get_stock_quote(ticker)
            if quote:
                context["stock_data"] = {
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                    "volume": quote.volume
                }
        
            # For technical analysis requests, get indicators
            if 'technical_analysis' in intent_analysis.get('all_intents', []):
                historical_data = self.market_service.get_historical_data(ticker, "6mo", "1d")
                if not historical_data.get("empty"):
                    df = self.market_service._dict_to_df(historical_data)
                    prediction = self.prediction_service.predict_stock_price(df, ticker)
                
                    if prediction and prediction.technical_indicators:
                        context["technical_indicators"] = {
                            "rsi": prediction.technical_indicators.rsi,
                            "macd": prediction.technical_indicators.macd,
                            "sma_20": prediction.technical_indicators.sma_20,
                            "sma_50": prediction.technical_indicators.sma_50
                        }
        
            # For news requests, get sentiment
            if 'news_sentiment' in intent_analysis.get('all_intents', []):
                sentiment = await self.news_service.analyze_sentiment(ticker, hours_back=24)
                if sentiment:
                    context["sentiment"] = {
                        "overall_sentiment": sentiment.overall_sentiment.value,
                        "sentiment_score": sentiment.sentiment_score,
                        "total_articles": sentiment.total_articles
                    }
    
        except Exception as e:
            print(f"Error gathering context: {e}")
    
        return context if context else None

    async def _get_llm_response(
    self,
    message: str,
    ticker: Optional[str],
    context: Optional[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Get response from LLM with context"""
        if not self.llm_enabled or not self.llm_service:
            return None
    
        try:
            # Convert conversation history to Message objects
            messages = []
            if conversation_history:
                for entry in conversation_history[-5:]:  # Last 5 messages
                    if entry.get("role") == "user":
                        messages.append(Message(ConversationRole.USER, entry.get("content", "")))
                    elif entry.get("role") == "assistant":
                        messages.append(Message(ConversationRole.ASSISTANT, entry.get("content", "")))
        
            # Add current message
            messages.append(Message(ConversationRole.USER, message))
        
            # Get LLM response
            response = await self.llm_service.chat(
                messages=messages,
                context=context
            )
        
            return response
        
        except Exception as e:
            print(f"LLM error: {e}")
            return None
    
    async def generate_response(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """Generate AI-powered response"""
        try:
            # Extract ticker if not provided
            ticker = request.ticker or self.extract_ticker_from_message(request.message)
            
            # Determine user intent
            intent_analysis = self.determine_intent(request.message, ticker)
            primary_intent = intent_analysis['primary_intent']
            
            # Create explanation context
            context = ExplanationContext(
                user_experience_level=ComplexityLevel(request.complexity_level),
                preferred_tone=ToneStyle(request.tone),
                include_educational=request.complexity_level == "beginner"
            )

            if hasattr(request, 'use_llm') and request.use_llm and self.llm_enabled:
            # Gather context for LLM
                llm_context = await self._gather_context_for_llm(ticker, intent_analysis)
            
            # Get LLM response
            llm_response = await self._get_llm_response(
                request.message,
                ticker,
                llm_context,
                request.conversation_history
            )
            
            if llm_response:
                # LLM provided a response! Use it
                return EnhancedChatResponse(
                    message=llm_response,
                    ticker=ticker,
                    response_type=primary_intent,
                    analytics=llm_context if llm_context else {},
                    suggestions=self._generate_suggestions(ticker, primary_intent),
                    confidence_score=0.9,
                    data_sources=["OpenAI GPT", "Market Data"] if llm_context else ["OpenAI GPT"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Route to appropriate handler based on intent
            if primary_intent == 'greeting':
                return await self._handle_greeting(request, context)
            
            elif primary_intent == 'stock_analysis' and ticker:
                return await self._handle_stock_analysis(request, ticker, context)
            
            elif primary_intent == 'buy_sell_advice' and ticker:
                return await self._handle_trading_advice(request, ticker, context)
            
            elif primary_intent == 'price_prediction' and ticker:
                return await self._handle_price_prediction(request, ticker, context)
            
            elif primary_intent == 'news_sentiment' and ticker:
                return await self._handle_news_analysis(request, ticker, context)
            
            elif primary_intent == 'technical_analysis' and ticker:
                return await self._handle_technical_analysis(request, ticker, context)
            
            elif primary_intent == 'educational':
                return await self._handle_educational(request, context)
            
            elif primary_intent == 'market_overview':
                return await self._handle_market_overview(request, context)
            
            elif primary_intent == 'comparison' and ticker:
                return await self._handle_comparison(request, ticker, context)
            
            else:
                return await self._handle_general(request, ticker, context, intent_analysis)
                
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I'm having trouble processing that request. Could you try rephrasing your question?",
                response_type="error",
                analytics={"error": str(e)},
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    def _generate_suggestions(self, ticker: Optional[str], intent: str) -> List[str]:
        """Generate contextual suggestions"""
        if ticker:
            return [
                f"What's the technical analysis for {ticker}?",
                f"Show me {ticker}'s news",
                f"Should I buy {ticker}?",
                f"Compare {ticker} to another stock"
         ]
        else:
            return [
                "Analyze a stock (e.g., 'Analyze AAPL')",
                "Ask about market conditions",
                "Learn about investing concepts",
                "Get trading advice"
            ]


    async def _handle_greeting(self, request: EnhancedChatRequest, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle greeting messages"""
        greeting_responses = {
            ToneStyle.CONVERSATIONAL: "Hi! I'm your AI financial advisor. I can help you analyze stocks, explain market movements, provide trading insights, and answer investment questions. What would you like to know?",
            ToneStyle.PROFESSIONAL: "Welcome to your AI Financial Advisory Service. I provide comprehensive investment analysis, market insights, and portfolio guidance. How may I assist you today?",
            ToneStyle.EDUCATIONAL: "Hello! I'm here to help you learn about investing and analyze the markets. I can explain complex financial concepts in simple terms and provide detailed stock analysis. What would you like to explore?"
        }
        
        message = greeting_responses.get(context.preferred_tone, greeting_responses[ToneStyle.CONVERSATIONAL])
        
        suggestions = [
            "Analyze a specific stock (e.g., 'Analyze AAPL')",
            "Ask for trading advice (e.g., 'Should I buy TSLA?')",
            "Get market overview (e.g., 'How is the market doing?')",
            "Learn about investing (e.g., 'Explain technical analysis')"
        ]
        
        return EnhancedChatResponse(
            message=message,
            response_type="greeting",
            suggestions=suggestions,
            confidence_score=1.0,
            data_sources=["AI Assistant"],
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def _handle_stock_analysis(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle comprehensive stock analysis requests"""
        try:
            # Generate comprehensive explanation
            explanation = await self.explanation_service.explain_stock_analysis(ticker, context)
            
            if not explanation:
                return EnhancedChatResponse(
                    message=f"I couldn't analyze {ticker} right now. The stock might not exist or data might be unavailable. Please check the ticker symbol and try again.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Generate conversational response
            chat_message = ChatResponseGenerator.generate_stock_chat_response(explanation, request.message)
            
            # Extract analytics for frontend
            analytics = {
                "stock_data": {
                    "symbol": explanation.symbol,
                    "analysis_type": explanation.explanation_type.value,
                    "confidence": explanation.confidence_score
                },
                "key_insights": [
                    {
                        "category": insight.category,
                        "title": insight.title,
                        "importance": insight.importance
                    }
                    for insight in explanation.key_insights[:3]
                ],
                "recommendations": explanation.recommendations[:2],
                "risk_level": "moderate"  # Could be derived from insights
            }
            
            suggestions = [
                f"Tell me more about {ticker}'s technical indicators",
                f"What's the latest news on {ticker}?",
                f"Should I buy {ticker} now?",
                "Compare this with another stock"
            ]
            
            return EnhancedChatResponse(
                message=chat_message,
                ticker=ticker,
                response_type="stock_analysis",
                analytics=analytics,
                suggestions=suggestions,
                confidence_score=explanation.confidence_score,
                data_sources=explanation.data_sources,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I encountered an issue analyzing {ticker}. Let me try a basic analysis instead.",
                response_type="error",
                ticker=ticker,
                analytics={"error": str(e)},
                confidence_score=0.2,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_trading_advice(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle buy/sell/hold advice requests"""
        try:
            # Get comprehensive data for trading advice
            stock_data = await self.data_collector.collect_stock_data(ticker, include_historical=True)
            
            if not stock_data or "error" in stock_data:
                return EnhancedChatResponse(
                    message=f"I couldn't get current data for {ticker} to provide trading advice. Please try again later.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Generate trading signal explanation
            explanation = await self.explanation_service.explain_stock_analysis(ticker, context)
            
            if explanation:
                # Extract trading recommendation from insights
                trading_insights = [insight for insight in explanation.key_insights if 'signal' in insight.title.lower() or 'buy' in insight.description.lower() or 'sell' in insight.description.lower()]
                
                if trading_insights:
                    primary_insight = trading_insights[0]
                    
                    # Generate personalized advice
                    advice_templates = {
                        ToneStyle.CONVERSATIONAL: f"Based on my analysis of {ticker}, here's what I think: {primary_insight.description} ",
                        ToneStyle.PROFESSIONAL: f"Our analysis of {ticker} indicates: {primary_insight.description} ",
                        ToneStyle.EDUCATIONAL: f"Let me explain the trading outlook for {ticker}: {primary_insight.description} "
                    }
                    
                    base_message = advice_templates.get(context.preferred_tone, advice_templates[ToneStyle.CONVERSATIONAL])
                    
                    # Add risk warning
                    risk_warning = "Remember, this is analysis based on current data and market conditions can change quickly. Always consider your risk tolerance and investment goals."
                    
                    full_message = base_message + f"\n\n⚠️ {risk_warning}"
                    
                    # Add key recommendations
                    if explanation.recommendations:
                        full_message += f"\n\nKey recommendations:\n• {explanation.recommendations[0]}"
                        if len(explanation.recommendations) > 1:
                            full_message += f"\n• {explanation.recommendations[1]}"
                    
                    analytics = {
                        "trading_signal": primary_insight.title,
                        "confidence": primary_insight.confidence,
                        "risk_level": "moderate",  # Could be extracted from insights
                        "key_factors": explanation.recommendations[:2]
                    }
                    
                    suggestions = [
                        f"What are the risks of investing in {ticker}?",
                        f"Show me {ticker}'s technical analysis",
                        f"What's the price target for {ticker}?",
                        "Help me with position sizing"
                    ]
                    
                    return EnhancedChatResponse(
                        message=full_message,
                        ticker=ticker,
                        response_type="trading_advice",
                        analytics=analytics,
                        suggestions=suggestions,
                        confidence_score=primary_insight.confidence,
                        data_sources=explanation.data_sources,
                        timestamp=datetime.utcnow().isoformat()
                    )
            
            # Fallback to basic advice
            quote_data = stock_data.get("quote")
            if quote_data:
                price = quote_data.get("price", 0)
                change_percent = quote_data.get("change_percent", 0)
                
                if change_percent > 5:
                    message = f"{ticker} is up {change_percent:.1f}% today to ${price:.2f}. Large moves like this can be risky - consider waiting for a pullback or using smaller position sizes."
                elif change_percent < -5:
                    message = f"{ticker} is down {change_percent:.1f}% today to ${price:.2f}. This could be a buying opportunity if the fundamentals are strong, but make sure to research the reason for the decline."
                else:
                    message = f"{ticker} is trading at ${price:.2f} with {change_percent:+.1f}% movement today. The price action appears relatively stable for analysis."
                
                return EnhancedChatResponse(
                    message=message,
                    ticker=ticker,
                    response_type="trading_advice",
                    analytics={"price": price, "change_percent": change_percent},
                    confidence_score=0.6,
                    data_sources=["Market Data"],
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            pass
        
        return EnhancedChatResponse(
            message=f"I'm having trouble getting trading advice for {ticker} right now. Please try again or ask about a different stock.",
            response_type="error",
            ticker=ticker,
            confidence_score=0.0,
            data_sources=["Error Handler"],
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def _handle_price_prediction(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle price prediction requests"""
        try:
            # Get historical data for prediction
            historical_data = self.market_service.get_historical_data(ticker, "1y", "1d")
            
            if historical_data.get("empty"):
                return EnhancedChatResponse(
                    message=f"I couldn't get historical data for {ticker} to make a price prediction.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            df = self.market_service._dict_to_df(historical_data)
            prediction = self.prediction_service.predict_stock_price(df, ticker)
            
            if not prediction:
                return EnhancedChatResponse(
                    message=f"I couldn't generate a price prediction for {ticker} right now.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Generate conversational prediction response
            direction = "up" if prediction.price_change_percent > 0 else "down"
            confidence_desc = "high" if prediction.confidence_score > 0.8 else "moderate" if prediction.confidence_score > 0.6 else "low"
            
            message = f"Based on my {prediction.model_used} analysis, {ticker} could move {direction} to around ${prediction.predicted_price:.2f} ({prediction.price_change_percent:+.1f}%) in the near term.\n\n"
            message += f"Current price: ${prediction.current_price:.2f}\n"
            message += f"Predicted price: ${prediction.predicted_price:.2f}\n"
            message += f"Confidence: {confidence_desc} ({prediction.confidence_score:.0%})\n\n"
            
            if prediction.support_levels:
                message += f"Key support level: ${prediction.support_levels[0]:.2f}\n"
            if prediction.resistance_levels:
                message += f"Key resistance level: ${prediction.resistance_levels[0]:.2f}\n"
            
            message += f"\n⚠️ This is a short-term prediction based on technical analysis. Market conditions can change rapidly."
            
            analytics = {
                "prediction": {
                    "current_price": prediction.current_price,
                    "predicted_price": prediction.predicted_price,
                    "change_percent": prediction.price_change_percent,
                    "confidence": prediction.confidence_score,
                    "model": prediction.model_used,
                    "trend": prediction.trend_direction.value
                },
                "levels": {
                    "support": prediction.support_levels[:2] if prediction.support_levels else [],
                    "resistance": prediction.resistance_levels[:2] if prediction.resistance_levels else []
                }
            }
            
            suggestions = [
                f"What factors drive {ticker}'s price?",
                f"Show me {ticker}'s technical analysis",
                f"Is {ticker} a good long-term investment?",
                "Explain how you make predictions"
            ]
            
            return EnhancedChatResponse(
                message=message,
                ticker=ticker,
                response_type="price_prediction",
                analytics=analytics,
                suggestions=suggestions,
                confidence_score=prediction.confidence_score,
                data_sources=["Technical Analysis", "ML Models"],
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I encountered an issue predicting {ticker}'s price. Please try again later.",
                response_type="error",
                ticker=ticker,
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_news_analysis(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle news and sentiment analysis requests"""
        try:
            sentiment_analysis = await self.news_service.analyze_sentiment(ticker, hours_back=24)
            
            if not sentiment_analysis:
                return EnhancedChatResponse(
                    message=f"I couldn't find recent news for {ticker} to analyze sentiment.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Generate news sentiment response
            sentiment_desc = sentiment_analysis.overall_sentiment.value.lower()
            
            message = f"Recent news sentiment for {ticker} appears {sentiment_desc} based on {sentiment_analysis.total_articles} articles I analyzed.\n\n"
            message += f"Sentiment score: {sentiment_analysis.sentiment_score:+.2f}\n"
            message += f"Article breakdown: {sentiment_analysis.positive_articles} positive, {sentiment_analysis.negative_articles} negative, {sentiment_analysis.neutral_articles} neutral\n\n"
            
            if sentiment_analysis.trending_keywords:
                message += f"Key topics: {', '.join(sentiment_analysis.trending_keywords[:5])}\n\n"
            
            if sentiment_analysis.recent_breaking_news:
                message += f"⚠️ {len(sentiment_analysis.recent_breaking_news)} breaking news articles detected - expect potential volatility.\n\n"
            
            if sentiment_analysis.sentiment_trend != "stable":
                message += f"Sentiment trend: {sentiment_analysis.sentiment_trend} over time\n"
            
            analytics = {
                "sentiment": {
                    "overall": sentiment_analysis.overall_sentiment.value,
                    "score": sentiment_analysis.sentiment_score,
                    "confidence": sentiment_analysis.confidence,
                    "trend": sentiment_analysis.sentiment_trend
                },
                "articles": {
                    "total": sentiment_analysis.total_articles,
                    "positive": sentiment_analysis.positive_articles,
                    "negative": sentiment_analysis.negative_articles,
                    "breaking": len(sentiment_analysis.recent_breaking_news)
                },
                "keywords": sentiment_analysis.trending_keywords[:5]
            }
            
            suggestions = [
                f"How might this news affect {ticker}'s price?",
                f"Should I buy {ticker} based on this news?",
                f"Show me {ticker}'s technical analysis",
                "Explain what drives stock prices"
            ]
            
            return EnhancedChatResponse(
                message=message,
                ticker=ticker,
                response_type="news_sentiment",
                analytics=analytics,
                suggestions=suggestions,
                confidence_score=sentiment_analysis.confidence,
                data_sources=["News Analysis", "Sentiment Analysis"],
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I had trouble analyzing news for {ticker}. Please try again.",
                response_type="error",
                ticker=ticker,
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_technical_analysis(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle technical analysis requests"""
        try:
            # Get historical data
            historical_data = self.market_service.get_historical_data(ticker, "6mo", "1d")
            
            if historical_data.get("empty"):
                return EnhancedChatResponse(
                    message=f"I couldn't get historical data for {ticker} to perform technical analysis.",
                    response_type="error",
                    ticker=ticker,
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            df = self.market_service._dict_to_df(historical_data)
            prediction = self.prediction_service.predict_stock_price(df, ticker)
            
            if not prediction or not prediction.technical_indicators:
                # Fallback to basic technical analysis (your original logic)
                df_basic = to_df(fetch_historical_data(ticker))
                if not df_basic.empty:
                    df_basic = add_indicators(df_basic)
                    
                    last_rsi = float(df_basic['RSI'].iloc[-1])
                    last_macd = float(df_basic['MACD'].iloc[-1])
                    last_signal = float(df_basic['Signal_Line'].iloc[-1])
                    
                    message = f"Technical analysis for {ticker}:\n\n"
                    message += f"RSI: {last_rsi:.1f} ({'Oversold' if last_rsi < 30 else 'Overbought' if last_rsi > 70 else 'Neutral'})\n"
                    message += f"MACD: {last_macd:.4f}\n"
                    message += f"Signal Line: {last_signal:.4f}\n"
                    message += f"MACD Signal: {'Bullish' if last_macd > last_signal else 'Bearish'}\n\n"
                    
                    if context.include_educational:
                        message += "📚 RSI measures overbought/oversold conditions (0-100 scale)\n"
                        message += "📚 MACD shows momentum by comparing moving averages"
                    
                    return EnhancedChatResponse(
                        message=message,
                        ticker=ticker,
                        response_type="technical_analysis",
                        analytics={"RSI": last_rsi, "MACD": last_macd, "signal_line": last_signal},
                        confidence_score=0.7,
                        data_sources=["Technical Analysis"],
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    raise Exception("No data available")
            
            indicators = prediction.technical_indicators
            
            # Generate comprehensive technical analysis response
            message = f"Technical analysis for {ticker}:\n\n"
            
            # RSI Analysis
            rsi_condition = "oversold" if indicators.rsi < 30 else "overbought" if indicators.rsi > 70 else "neutral"
            message += f"📊 RSI: {indicators.rsi:.1f} ({rsi_condition})\n"
            
            # MACD Analysis
            macd_signal = "bullish" if indicators.macd > indicators.macd_signal else "bearish"
            message += f"📊 MACD: {indicators.macd:.4f} ({macd_signal} signal)\n"
            
            # Moving Averages
            if indicators.sma_20 and indicators.sma_50:
                current_price = prediction.current_price
                ma_trend = "upward" if current_price > indicators.sma_20 > indicators.sma_50 else "downward" if current_price < indicators.sma_20 < indicators.sma_50 else "sideways"
                message += f"📊 Trend: {ma_trend} (price vs moving averages)\n"
                message += f"📊 20-day MA: ${indicators.sma_20:.2f}\n"
                message += f"📊 50-day MA: ${indicators.sma_50:.2f}\n"
            
            # Support/Resistance
            if prediction.support_levels and prediction.resistance_levels:
                message += f"📊 Support: ${prediction.support_levels[0]:.2f}\n"
                message += f"📊 Resistance: ${prediction.resistance_levels[0]:.2f}\n"
            
            # Volatility
            message += f"📊 Volatility: {prediction.volatility_score:.2f} ({'High' if prediction.volatility_score > 0.3 else 'Low' if prediction.volatility_score < 0.15 else 'Moderate'})\n"
            
            # Bollinger Bands
            if indicators.bb_upper and indicators.bb_lower:
                bb_position = (prediction.current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
                bb_desc = "near upper band" if bb_position > 0.8 else "near lower band" if bb_position < 0.2 else "middle range"
                message += f"📊 Bollinger Bands: {bb_desc}\n"
            
            if context.include_educational:
                message += "\n📚 Educational Notes:\n"
                message += "• RSI below 30 = potentially oversold, above 70 = potentially overbought\n"
                message += "• MACD above signal line = bullish momentum\n"
                message += "• Price above moving averages = uptrend\n"
                message += "• Support = price level where buying interest emerges\n"
                message += "• Resistance = price level where selling pressure increases"
            
            analytics = {
                "technical_indicators": {
                    "rsi": indicators.rsi,
                    "macd": indicators.macd,
                    "macd_signal": indicators.macd_signal,
                    "sma_20": indicators.sma_20,
                    "sma_50": indicators.sma_50,
                    "trend": prediction.trend_direction.value,
                    "volatility": prediction.volatility_score
                },
                "levels": {
                    "support": prediction.support_levels[:2] if prediction.support_levels else [],
                    "resistance": prediction.resistance_levels[:2] if prediction.resistance_levels else []
                }
            }
            
            suggestions = [
                f"What do these indicators mean for {ticker}?",
                f"Should I buy {ticker} based on technicals?",
                f"How reliable are technical indicators?",
                "Explain Bollinger Bands to me"
            ]
            
            return EnhancedChatResponse(
                message=message,
                ticker=ticker,
                response_type="technical_analysis",
                analytics=analytics,
                suggestions=suggestions,
                confidence_score=0.8,
                data_sources=["Technical Analysis", "Market Data"],
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I couldn't perform technical analysis on {ticker} right now. Please try again.",
                response_type="error",
                ticker=ticker,
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_educational(self, request: EnhancedChatRequest, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle educational requests"""
        from app.services.explanation_services import EducationalContentGenerator
        
        message_lower = request.message.lower()
        
        # Map common educational topics
        topic_mapping = {
            'technical analysis': 'technical_analysis',
            'fundamental analysis': 'fundamental_analysis',
            'diversification': 'diversification',
            'risk management': 'risk_management',
            'rsi': 'technical_analysis',
            'macd': 'technical_analysis',
            'moving averages': 'technical_analysis',
            'portfolio': 'diversification',
            'risk': 'risk_management'
        }
        
        # Find matching topic
        topic = None
        for key, value in topic_mapping.items():
            if key in message_lower:
                topic = value
                break
        
        if topic:
            educational_content = EducationalContentGenerator.explain_concept(
                topic, context.user_experience_level
            )
            
            # Add practical examples
            if topic == 'technical_analysis':
                educational_content += "\n\n💡 Practical tip: Start with simple indicators like RSI and moving averages before exploring complex strategies."
            elif topic == 'diversification':
                educational_content += "\n\n💡 Practical tip: A good rule of thumb is to limit individual positions to 5-10% of your portfolio."
            elif topic == 'risk_management':
                educational_content += "\n\n💡 Practical tip: Never invest more than you can afford to lose, and always use stop-losses for protection."
            
            suggestions = [
                "Give me an example with a real stock",
                "How do I apply this in practice?",
                "What are common mistakes to avoid?",
                "Explain another investing concept"
            ]
            
            return EnhancedChatResponse(
                message=educational_content,
                response_type="educational",
                suggestions=suggestions,
                confidence_score=0.9,
                data_sources=["Educational Content"],
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            # General educational response
            message = "I'd be happy to explain investing concepts! I can help you understand:\n\n"
            message += "📚 Technical Analysis - Reading price charts and indicators\n"
            message += "📚 Fundamental Analysis - Company valuation and financials\n"
            message += "📚 Risk Management - Protecting your investments\n"
            message += "📚 Diversification - Spreading investment risk\n"
            message += "📚 Portfolio Management - Building and maintaining investments\n\n"
            message += "What would you like to learn about?"
            
            suggestions = [
                "Explain technical analysis",
                "What is diversification?",
                "How do I manage risk?",
                "Teach me about P/E ratios"
            ]
            
            return EnhancedChatResponse(
                message=message,
                response_type="educational",
                suggestions=suggestions,
                confidence_score=0.8,
                data_sources=["Educational Content"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_market_overview(self, request: EnhancedChatRequest, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle market overview requests"""
        try:
            market_explanation = await self.explanation_service.generate_market_explanation(context)
            
            if market_explanation:
                # Generate conversational market summary
                message = market_explanation.summary + "\n\n"
                
                # Add key insights
                if market_explanation.key_insights:
                    message += "Key market insights:\n"
                    for insight in market_explanation.key_insights[:3]:
                        message += f"• {insight.description}\n"
                    message += "\n"
                
                # Add recommendations
                if market_explanation.recommendations:
                    message += "Market recommendations:\n"
                    for rec in market_explanation.recommendations[:2]:
                        message += f"• {rec}\n"
                
                analytics = {
                    "market_status": "active",  # Would come from market data
                    "sentiment": "mixed",  # Would come from market analysis
                    "key_themes": []  # Would come from insights
                }
                
                suggestions = [
                    "Which sectors are performing well?",
                    "Should I invest in this market?",
                    "What are the biggest risks right now?",
                    "Analyze a specific stock for me"
                ]
                
                return EnhancedChatResponse(
                    message=message,
                    response_type="market_overview",
                    analytics=analytics,
                    suggestions=suggestions,
                    confidence_score=market_explanation.confidence_score,
                    data_sources=market_explanation.data_sources,
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                # Fallback market overview
                message = "The market is currently active with mixed sentiment across different sectors. "
                message += "Major indices are showing typical daily volatility. "
                message += "I recommend staying informed about economic indicators and maintaining a diversified portfolio."
                
                return EnhancedChatResponse(
                    message=message,
                    response_type="market_overview",
                    confidence_score=0.5,
                    data_sources=["General Market Knowledge"],
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            return EnhancedChatResponse(
                message="I'm having trouble getting current market information. Please try asking about a specific stock or topic instead.",
                response_type="error",
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_comparison(self, request: EnhancedChatRequest, ticker: str, context: ExplanationContext) -> EnhancedChatResponse:
        """Handle stock comparison requests"""
        # Extract second ticker from message
        tickers = re.findall(r'\b[A-Z]{1,5}\b', request.message.upper())
        
        if len(tickers) < 2:
            return EnhancedChatResponse(
                message="I'd be happy to compare stocks for you! Please provide two ticker symbols, like 'Compare AAPL vs MSFT' or 'TSLA versus AMZN'.",
                response_type="comparison",
                confidence_score=0.3,
                data_sources=["Chat Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
        
        ticker1, ticker2 = tickers[0], tickers[1]
        
        try:
            # Get basic data for both stocks
            data1 = await self.data_collector.collect_stock_data(ticker1)
            data2 = await self.data_collector.collect_stock_data(ticker2)
            
            if not data1 or not data2 or "error" in data1 or "error" in data2:
                return EnhancedChatResponse(
                    message=f"I couldn't get data to compare {ticker1} and {ticker2}. Please check the ticker symbols.",
                    response_type="error",
                    confidence_score=0.0,
                    data_sources=["Error Handler"],
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Extract comparison metrics
            quote1 = data1.get("quote", {})
            quote2 = data2.get("quote", {})
            
            message = f"Comparison between {ticker1} and {ticker2}:\n\n"
            
            # Price comparison
            price1 = quote1.get("price", 0)
            price2 = quote2.get("price", 0)
            change1 = quote1.get("change_percent", 0)
            change2 = quote2.get("change_percent", 0)
            
            message += f"📊 Current Prices:\n"
            message += f"{ticker1}: ${price1:.2f} ({change1:+.2f}%)\n"
            message += f"{ticker2}: ${price2:.2f} ({change2:+.2f}%)\n\n"
            
            # Performance comparison
            better_performer = ticker1 if change1 > change2 else ticker2
            message += f"📈 Today's Performance: {better_performer} is outperforming\n\n"
            
            # Volume comparison if available
            vol1 = quote1.get("volume", 0)
            vol2 = quote2.get("volume", 0)
            if vol1 and vol2:
                higher_volume = ticker1 if vol1 > vol2 else ticker2
                message += f"📊 Trading Volume: {higher_volume} has higher volume today\n\n"
            
            message += "For a detailed comparison, I recommend looking at:\n"
            message += "• Financial metrics (P/E, revenue growth, debt levels)\n"
            message += "• Technical indicators and trends\n"
            message += "• Business fundamentals and competitive position\n"
            message += "• Risk factors and volatility"
            
            analytics = {
                "comparison": {
                    ticker1: {"price": price1, "change": change1, "volume": vol1},
                    ticker2: {"price": price2, "change": change2, "volume": vol2}
                },
                "winner": {"performance": better_performer}
            }
            
            suggestions = [
                f"Analyze {ticker1} in detail",
                f"Analyze {ticker2} in detail", 
                f"Which is better for long-term investing?",
                "Show me technical analysis for both"
            ]
            
            return EnhancedChatResponse(
                message=message,
                ticker=f"{ticker1},{ticker2}",
                response_type="comparison",
                analytics=analytics,
                suggestions=suggestions,
                confidence_score=0.7,
                data_sources=["Market Data", "Comparison Analysis"],
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return EnhancedChatResponse(
                message=f"I had trouble comparing {ticker1} and {ticker2}. Please try again or ask about individual stocks.",
                response_type="error",
                confidence_score=0.0,
                data_sources=["Error Handler"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _handle_general(self, request: EnhancedChatRequest, ticker: Optional[str], context: ExplanationContext, intent_analysis: Dict) -> EnhancedChatResponse:
        """Handle general questions and unclear requests"""
        
        if ticker:
            # General question about a specific stock
            message = f"I can help you with {ticker}! What would you like to know? I can provide:\n\n"
            message += f"📊 Complete stock analysis\n"
            message += f"📈 Price predictions and technical analysis\n"
            message += f"📰 Latest news and sentiment analysis\n"
            message += f"💡 Buy/sell/hold recommendations\n"
            message += f"🎯 Trading signals and price targets\n\n"
            message += f"Just ask me something like 'Analyze {ticker}' or 'Should I buy {ticker}?'"
            
            suggestions = [
                f"Analyze {ticker}",
                f"Should I buy {ticker}?",
                f"What's {ticker}'s price prediction?",
                f"Show me {ticker}'s news sentiment"
            ]
        else:
            # General financial question
            message = "I'm your AI financial advisor and I'm here to help! I can assist you with:\n\n"
            message += "📊 Stock Analysis - Get comprehensive analysis of any stock\n"
            message += "📈 Trading Advice - Buy/sell/hold recommendations with reasoning\n"
            message += "🔮 Price Predictions - ML-powered price forecasts\n"
            message += "📰 News & Sentiment - Latest news impact analysis\n"
            message += "📚 Education - Learn about investing concepts\n"
            message += "🌍 Market Overview - Current market conditions\n\n"
            message += "Try asking me about a specific stock (like 'Analyze AAPL') or an investing topic!"
            
            suggestions = [
                "Analyze AAPL",
                "What's happening in the market?",
                "Teach me about technical analysis",
                "Should I buy TSLA?"
            ]
        
        return EnhancedChatResponse(
            message=message,
            ticker=ticker,
            response_type="general",
            suggestions=suggestions,
            confidence_score=0.6,
            data_sources=["AI Assistant"],
            timestamp=datetime.utcnow().isoformat()
        )

# Initialize chat service
chat_service = ChatService()

# ENHANCED ENDPOINTS

@router.post("/message", response_model=EnhancedChatResponse)
async def enhanced_chat(request: EnhancedChatRequest):
    """
    Enhanced AI chat endpoint with comprehensive financial analysis capabilities
    """
    try:
        response = await chat_service.generate_response(request)
        
        # Store conversation history if user_id provided
        if request.user_id:
            conversation_key = f"chat_history_{request.user_id}"
            
            # Get existing history
            existing_history = chat_service.cache_service.get(
                CacheType.CHAT_HISTORY, conversation_key
            ) or []
            
            # Add new conversation
            conversation_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": request.message,
                "ai_response": response.message,
                "ticker": response.ticker,
                "response_type": response.response_type
            }
            
            existing_history.append(conversation_entry)
            
            # Keep only last 50 messages
            if len(existing_history) > 50:
                existing_history = existing_history[-50:]
            
            # Cache updated history
            chat_service.cache_service.set(
                CacheType.CHAT_HISTORY, 
                conversation_key, 
                existing_history, 
                ttl=604800  # 1 week
            )
        
        return response
        
    except Exception as e:
        return EnhancedChatResponse(
            message="I'm experiencing some technical difficulties. Please try again in a moment.",
            response_type="error",
            confidence_score=0.0,
            data_sources=["Error Handler"],
            timestamp=datetime.utcnow().isoformat()
        )

@router.get("/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = Query(20, le=50)):
    """Get user's chat history"""
    try:
        conversation_key = f"chat_history_{user_id}"
        history = chat_service.cache_service.get(CacheType.CHAT_HISTORY, conversation_key) or []
        
        # Return recent conversations
        recent_history = history[-limit:] if len(history) > limit else history
        
        return {
            "user_id": user_id,
            "total_conversations": len(history),
            "returned_conversations": len(recent_history),
            "history": recent_history,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@router.delete("/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear user's chat history"""
    try:
        conversation_key = f"chat_history_{user_id}"
        success = chat_service.cache_service.delete(CacheType.CHAT_HISTORY, conversation_key)
        
        return {
            "user_id": user_id,
            "cleared": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.post("/analyze", response_model=EnhancedChatResponse)
async def chat_stock_analysis(request: ChatAnalyticsRequest):
    """
    Dedicated endpoint for stock analysis through chat interface
    """
    try:
        # Create chat request for analysis
        chat_request = EnhancedChatRequest(
            message=f"Analyze {request.ticker}",
            ticker=request.ticker,
            complexity_level="intermediate",
            tone="conversational"
        )
        
        if request.analysis_type == "comprehensive":
            chat_request.message = f"Give me a comprehensive analysis of {request.ticker}"
        elif request.analysis_type == "signals":
            chat_request.message = f"What are the trading signals for {request.ticker}?"
        elif request.analysis_type == "explanation":
            chat_request.message = f"Explain {request.ticker}'s current situation"
        
        response = await chat_service.generate_response(chat_request)
        return response
        
    except Exception as e:
        return EnhancedChatResponse(
            message=f"I couldn't analyze {request.ticker} right now. Please try again.",
            response_type="error",
            confidence_score=0.0,
            data_sources=["Error Handler"],
            timestamp=datetime.utcnow().isoformat()
        )

@router.websocket("/stream/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    active_conversations[user_id] = websocket
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Create chat request
            chat_request = EnhancedChatRequest(
                message=message_data.get("message", ""),
                ticker=message_data.get("ticker"),
                user_id=user_id,
                complexity_level=message_data.get("complexity", "intermediate"),
                tone=message_data.get("tone", "conversational")
            )
            
            # Generate response
            response = await chat_service.generate_response(chat_request)
            
            # Send response
            await websocket.send_text(json.dumps({
                "type": "chat_response",
                "data": response.dict()
            }))
            
    except WebSocketDisconnect:
        if user_id in active_conversations:
            del active_conversations[user_id]
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Connection error occurred"
        }))
        if user_id in active_conversations:
            del active_conversations[user_id]

@router.get("/suggestions/{ticker}")
async def get_chat_suggestions(ticker: str):
    """Get suggested questions for a ticker"""
    suggestions = [
        f"Analyze {ticker}",
        f"Should I buy {ticker}?",
        f"What's {ticker}'s price prediction?",
        f"Show me {ticker}'s technical analysis",
        f"What's the latest news on {ticker}?",
        f"Is {ticker} a good long-term investment?",
        f"What are the risks of investing in {ticker}?",
        f"Compare {ticker} to its competitors"
    ]
    
    return {
        "ticker": ticker,
        "suggestions": suggestions,
        "timestamp": datetime.utcnow().isoformat()
    }

# LEGACY COMPATIBILITY ENDPOINT
@router.post("/legacy", response_model=ChatResponse)
def chat_with_ai_legacy(req: ChatRequest):
    """
    Legacy chat endpoint - exact replica of your original chat function
    Maintained for backward compatibility
    """
    payload = fetch_historical_data(req.ticker)
    df = to_df(payload)
    if df.empty:
        raise HTTPException(status_code=404, detail="Ticker not found")

    df = add_indicators(df)
    last_close = float(df["Close"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    if "buy" in req.question.lower():
        if rsi < 30:
            reply = f"RSI={rsi:.2f} (<30). Oversold: potential BUY."
            signal, sentiment = "Buy", "Bullish"
        elif rsi > 70:
            reply = f"RSI={rsi:.2f} (>70). Overbought: consider SELL/HOLD."
            signal, sentiment = "Sell", "Bearish"
        else:
            reply = f"RSI={rsi:.2f} neutral. Consider waiting for a clearer setup."
            signal, sentiment = "Hold", "Neutral"
    else:
        reply = f"Last close ${last_close:.2f}, RSI {rsi:.2f}. Ask about buy/sell outlook."
        signal, sentiment = "Neutral", "Neutral"

    return ChatResponse(
        ticker=req.ticker,
        question=req.question,
        reply=reply,
        analytics={
            "prediction": f"${last_close:.2f}",
            "signal": signal,
            "sentiment": sentiment,
        }
    )

@router.get("/health")
async def chat_health_check():
    """Health check for chat service"""
    try:
        # Test basic functionality
        test_response = await chat_service.generate_response(
            EnhancedChatRequest(message="Hello", complexity_level="intermediate")
        )
        
        return {
            "status": "healthy",
            "features": {
                "ai_responses": True,
                "stock_analysis": True,
                "technical_analysis": True,
                "news_sentiment": True,
                "educational_content": True,
                "websocket_support": True
            },
            "test_response_generated": bool(test_response),
            "active_conversations": len(active_conversations),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
    
@router.get("/llm/status")
async def llm_status():
    """Check if LLM is enabled and working"""
    return {
        "llm_enabled": chat_service.llm_enabled,
        "llm_available": chat_service.llm_service is not None,
        "model": chat_service.llm_service.model if chat_service.llm_service else None,
        "fallback_system": "rule-based + explanation services",
        "timestamp": datetime.utcnow().isoformat()
    }