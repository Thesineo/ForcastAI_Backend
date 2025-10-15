"""
LLM Service for AI Financial Advisor
OpenAI-powered natural language processing and conversation
"""

import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
from enum import Enum

from openai import AsyncOpenAI
from openai import OpenAIError

# Configure logging
logger = logging.getLogger(__name__)

class ConversationRole(Enum):
    """Conversation message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message:
    """Chat message structure"""
    def __init__(self, role: ConversationRole, content: str):
        self.role = role.value
        self.content = content
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }

class LLMService:
    """
    OpenAI-powered LLM service for financial advisory
    Handles conversation, context management, and financial analysis
    """
    
    # System prompts for different contexts
    FINANCIAL_ADVISOR_PROMPT = """You are an expert AI Financial Advisor with deep knowledge of:
- Stock market analysis and trading strategies
- Technical analysis indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.)
- Fundamental analysis (P/E ratios, earnings, financial statements, revenue growth)
- Risk management and portfolio diversification strategies
- Market trends, economic indicators, and macroeconomic factors
- Investment psychology and behavioral finance principles

Your role is to:
1. Provide clear, actionable financial advice tailored to the user's needs
2. Explain complex financial concepts in simple, understandable terms
3. Always include appropriate risk warnings and disclaimers
4. Base all recommendations on data, analysis, and sound financial principles
5. Consider the user's risk tolerance, investment goals, and time horizon
6. Be honest about uncertainty, limitations, and when professional advice is needed
7. Encourage responsible investing and continuous learning

IMPORTANT DISCLAIMERS:
- This is educational information, not professional financial advice
- Users should conduct their own research and due diligence
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- Consider consulting a licensed financial advisor for personalized guidance

Communication Style:
- Be conversational, helpful, and educational
- Maintain professionalism while being approachable
- Use analogies and examples to clarify complex concepts
- Provide specific, actionable insights when possible
- Acknowledge when you don't have enough information"""

    ANALYSIS_PROMPT = """You are a financial data analyst providing technical and fundamental analysis. Focus on:
- Interpreting technical indicators accurately
- Identifying chart patterns and trends
- Assessing risk factors and volatility
- Providing context from market conditions
- Making data-driven, objective observations
- Explaining the significance of key metrics

Be precise, analytical, and objective. Support your insights with specific data points when available."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize OpenAI LLM Service
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            model: Model to use (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"✅ OpenAI LLM service initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send chat messages and get response
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt (uses default if not provided)
            context: Optional additional context (market data, analysis, etc.)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
        
        Returns:
            LLM response text
        """
        try:
            # Prepare messages
            formatted_messages = self._format_messages(messages, system_prompt, context)
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            logger.info(f"LLM response generated successfully (tokens used: {response.usage.total_tokens})")
            return response_text
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"Failed to get LLM response: {str(e)}")
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            raise
    
    def _format_messages(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Format messages for OpenAI API"""
        
        formatted = []
        
        # Add system prompt
        system_content = system_prompt or self.FINANCIAL_ADVISOR_PROMPT
        
        # Add context to system prompt if provided
        if context:
            context_text = self._format_context(context)
            system_content += f"\n\n**Current Context and Data:**\n{context_text}"
        
        formatted.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation messages
        for msg in messages:
            formatted.append(msg.to_dict())
        
        return formatted
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context data into readable text for LLM"""
        
        context_parts = []
        
        # Stock data
        if "stock_data" in context:
            stock = context["stock_data"]
            context_parts.append("**Stock Information:**")
            context_parts.append(f"- Symbol: {stock.get('symbol', 'N/A')}")
            context_parts.append(f"- Current Price: ${stock.get('price', 'N/A')}")
            context_parts.append(f"- Change: {stock.get('change_percent', 'N/A')}%")
            if stock.get('volume'):
                context_parts.append(f"- Volume: {stock.get('volume'):,}")
            context_parts.append("")
        
        # Prediction data
        if "prediction" in context:
            pred = context["prediction"]
            context_parts.append("**AI Price Prediction:**")
            context_parts.append(f"- Predicted Price: ${pred.get('predicted_price', 'N/A')}")
            context_parts.append(f"- Expected Change: {pred.get('price_change_percent', 'N/A')}%")
            context_parts.append(f"- Confidence Level: {pred.get('confidence', 'N/A')}")
            context_parts.append("")
        
        # Technical indicators
        if "technical_indicators" in context:
            indicators = context["technical_indicators"]
            context_parts.append("**Technical Indicators:**")
            if indicators.get('rsi'):
                context_parts.append(f"- RSI: {indicators.get('rsi', 'N/A')}")
            if indicators.get('macd'):
                context_parts.append(f"- MACD: {indicators.get('macd', 'N/A')}")
            if indicators.get('sma_20'):
                context_parts.append(f"- 20-day SMA: ${indicators.get('sma_20', 'N/A')}")
            if indicators.get('sma_50'):
                context_parts.append(f"- 50-day SMA: ${indicators.get('sma_50', 'N/A')}")
            context_parts.append("")
        
        # Sentiment analysis
        if "sentiment" in context:
            sentiment = context["sentiment"]
            context_parts.append("**News Sentiment Analysis:**")
            context_parts.append(f"- Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
            context_parts.append(f"- Sentiment Score: {sentiment.get('sentiment_score', 'N/A')}")
            context_parts.append(f"- Articles Analyzed: {sentiment.get('total_articles', 'N/A')}")
            context_parts.append("")
        
        # User preferences
        if "user_preferences" in context:
            prefs = context["user_preferences"]
            context_parts.append("**User Profile:**")
            context_parts.append(f"- Risk Tolerance: {prefs.get('risk_tolerance', 'moderate').replace('_', ' ').title()}")
            context_parts.append(f"- Investment Timeframe: {prefs.get('timeframe', 'medium_term').replace('_', ' ').title()}")
            context_parts.append(f"- Experience Level: {prefs.get('experience_level', 'intermediate').title()}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def analyze_stock(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        prediction: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None
    ) -> str:
        """
        Analyze a stock and provide comprehensive insights
        
        Args:
            symbol: Stock ticker symbol
            stock_data: Current stock price and volume data
            prediction: AI price prediction data
            sentiment: News sentiment analysis
            user_query: Optional specific question from user
        
        Returns:
            Detailed stock analysis
        """
        
        # Build context
        context = {
            "stock_data": stock_data,
            "prediction": prediction,
            "sentiment": sentiment
        }
        
        # Create analysis query
        if user_query:
            query = f"Regarding {symbol}: {user_query}"
        else:
            query = f"Provide a comprehensive analysis of {symbol} including current situation, key insights, and investment considerations."
        
        messages = [
            Message(ConversationRole.USER, query)
        ]
        
        return await self.chat(
            messages=messages,
            system_prompt=self.ANALYSIS_PROMPT,
            context=context
        )
    
    async def explain_indicator(
        self,
        indicator_name: str,
        indicator_value: float,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Explain what a technical indicator means and its implications
        
        Args:
            indicator_name: Name of the indicator (e.g., "RSI", "MACD")
            indicator_value: Current value of the indicator
            symbol: Optional stock symbol for context
            context: Optional additional context
        
        Returns:
            Explanation of the indicator
        """
        
        symbol_text = f" for {symbol}" if symbol else ""
        query = f"Explain what a {indicator_name} value of {indicator_value}{symbol_text} means. What does this indicate about the stock's momentum and potential trading opportunities?"
        
        messages = [
            Message(ConversationRole.USER, query)
        ]
        
        return await self.chat(
            messages=messages,
            context=context,
            max_tokens=1000
        )
    
    async def generate_recommendation(
        self,
        symbol: str,
        analysis_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """
        Generate personalized investment recommendation
        
        Args:
            symbol: Stock ticker symbol
            analysis_data: Complete analysis data (price, prediction, sentiment, etc.)
            user_preferences: User's investment preferences and risk tolerance
        
        Returns:
            Personalized recommendation
        """
        
        context = {
            **analysis_data,
            "user_preferences": user_preferences
        }
        
        query = f"""Based on the current analysis of {symbol} and considering my investment profile, 
        should I buy, sell, or hold this stock? Please provide:
        1. A clear recommendation (Buy/Sell/Hold)
        2. Key reasons supporting your recommendation
        3. Important risks to consider
        4. Suggested entry/exit points if applicable"""
        
        messages = [
            Message(ConversationRole.USER, query)
        ]
        
        return await self.chat(
            messages=messages,
            context=context
        )
    
    async def answer_question(
        self,
        question: str,
        conversation_history: List[Message],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer a user question with full conversation context
        
        Args:
            question: User's question
            conversation_history: Previous conversation messages
            context: Optional additional context (market data, etc.)
        
        Returns:
            AI-generated answer
        """
        
        # Add new question to history
        messages = conversation_history + [
            Message(ConversationRole.USER, question)
        ]
        
        return await self.chat(
            messages=messages,
            context=context
        )
    
    async def summarize_market_news(
        self,
        news_items: List[Dict[str, Any]],
        focus_symbol: Optional[str] = None
    ) -> str:
        """
        Summarize market news and extract key insights
        
        Args:
            news_items: List of news articles with titles and summaries
            focus_symbol: Optional stock symbol to focus on
        
        Returns:
            News summary with key takeaways
        """
        
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', '')}"
            for item in news_items[:10]  # Limit to 10 articles
        ])
        
        focus_text = f" particularly regarding {focus_symbol}" if focus_symbol else ""
        
        query = f"""Summarize the following market news{focus_text}. 
        Identify key themes, important developments, and potential market impacts:

        {news_text}

        Provide:
        1. Main themes and trends
        2. Most significant developments
        3. Potential market implications
        4. Key takeaways for investors"""
        
        messages = [
            Message(ConversationRole.USER, query)
        ]
        
        return await self.chat(
            messages=messages,
            max_tokens=1500
        )


# Global service instance
_llm_service = None

def get_llm_service(
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> LLMService:
    """
    Get or create LLM service instance (Singleton pattern)
    
    Args:
        api_key: Optional OpenAI API key
        model: OpenAI model to use
    
    Returns:
        LLMService instance
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService(api_key=api_key, model=model)
        logger.info(f"Created LLM service instance with model: {model}")
    
    return _llm_service

def reset_llm_service():
    """Reset LLM service instance (useful for testing)"""
    global _llm_service
    _llm_service = None
    logger.info("Reset LLM service instance")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        """Test the LLM service"""
        print("🧪 Testing OpenAI LLM Service\n")
        
        try:
            # Initialize service
            service = get_llm_service()
            print("✅ LLM service initialized\n")
            
            # Test 1: Basic chat
            print("📝 Test 1: Basic Financial Question")
            messages = [
                Message(ConversationRole.USER, "What is RSI and how should I use it in trading?")
            ]
            response = await service.chat(messages)
            print(f"Response: {response[:200]}...\n")
            
            # Test 2: Stock analysis
            print("📊 Test 2: Stock Analysis")
            stock_data = {
                "symbol": "AAPL",
                "price": 178.50,
                "change_percent": 2.3,
                "volume": 52000000
            }
            analysis = await service.analyze_stock("AAPL", stock_data)
            print(f"Analysis: {analysis[:200]}...\n")
            
            # Test 3: Indicator explanation
            print("📈 Test 3: Indicator Explanation")
            explanation = await service.explain_indicator("RSI", 72.5, "AAPL")
            print(f"Explanation: {explanation[:200]}...\n")
            
            print("✅ All tests passed!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Run tests
    asyncio.run(test_llm())