"""
Enhanced analyze.py Router
Integrates with all new backend services while maintaining compatibility
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

# Your existing imports (kept for compatibility)
from app.db.db import SessionLocal, get_db
from app.db.model import AnalysisLog
from app.core.config import settings

# New service imports (your enhanced backend)
from app.services.market_data_services import get_market_service, MarketDataService
from app.services.prediction_services import get_prediction_service, PredictionService
from app.services.news_services import get_news_service, NewsService
from app.services.explanation_services import get_explanation_service, ExplanationService, ExplanationContext, ComplexityLevel, ToneStyle
from app.services.data_collector import get_data_collector, DataCollector
from app.services.cache import get_cache_service, CacheType

# Legacy imports (for backward compatibility)
from app.services.market_data_services import fetch_historical_data, to_df
from app.services.prediction_services import add_indicators
from app.services.prediction_services import forecast
from app.services.news_services import fetch_news
from app.services.prediction_services import generate_signal
from app.services.news_services import analyze_sentiment

router = APIRouter(prefix="/api/analyze", tags=["Analysis"])

# Pydantic models for new enhanced endpoints
class AnalysisRequest(BaseModel):
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    tone: str = "conversational"  # conversational, professional, educational
    include_news: bool = True
    include_predictions: bool = True
    include_explanation: bool = True

class ComprehensiveAnalysisResponse(BaseModel):
    ticker: str
    timestamp: str
    basic_analysis: Dict[str, Any]  # Your original response format
    enhanced_analysis: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    confidence_score: float
    data_sources: List[str]

class BatchAnalysisRequest(BaseModel):
    tickers: List[str]
    analysis_type: str = "basic"  # basic, comprehensive, signals_only

# Dependency injection for services
def get_market_data_service() -> MarketDataService:
    return get_market_service()

def get_prediction_service_dep() -> PredictionService:
    return get_prediction_service()

def get_news_service_dep() -> NewsService:
    return get_news_service()

def get_explanation_service_dep() -> ExplanationService:
    return get_explanation_service()

def get_data_collector_dep() -> DataCollector:
    return get_data_collector()

# ENHANCED ENDPOINTS

@router.get("/health")
async def health_check():
    """Health check endpoint for all analysis services"""
    try:
        market_service = get_market_service()
        prediction_service = get_prediction_service()
        news_service = get_news_service()
        explanation_service = get_explanation_service()
        
        health_checks = {
            "market_data": market_service.health_check(),
            "predictions": prediction_service.health_check(),
            "news": news_service.health_check(),
            "explanations": explanation_service.health_check(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        overall_status = "healthy" if all(
            check.get("status") == "healthy" 
            for check in health_checks.values() 
            if isinstance(check, dict)
        ) else "degraded"
        
        return {
            "status": overall_status,
            "services": health_checks
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/{ticker}")
async def analyze_stock_enhanced(
    ticker: str,
    horizon_days: int = Query(1, ge=1, le=30),
    model: str = Query("ensemble", regex="^(svr|prophet|ensemble|random_forest|xgboost)$"),
    complexity: str = Query("intermediate", regex="^(beginner|intermediate|advanced)$"),
    tone: str = Query("conversational", regex="^(conversational|professional|educational|confident|cautious)$"),
    include_explanation: bool = Query(True),
    include_news: bool = Query(True),
    db: Session = Depends(get_db),
    market_service: MarketDataService = Depends(get_market_data_service),
    prediction_service: PredictionService = Depends(get_prediction_service_dep),
    news_service: NewsService = Depends(get_news_service_dep),
    explanation_service: ExplanationService = Depends(get_explanation_service_dep)
):
    """
    Enhanced stock analysis endpoint - backward compatible with your original endpoint
    Now includes AI explanations, advanced ML predictions, and comprehensive analysis
    """
    try:
        # ORIGINAL ANALYSIS (for backward compatibility)
        data_payload = fetch_historical_data(ticker)
        df = to_df(data_payload)
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data found")

        df = add_indicators(df)

        # Your original prediction logic
        pred, model_used = forecast(df, horizon_days=horizon_days, model=model)
        
        last_close = float(df['Close'].iloc[-1])
        last_rsi = float(df['RSI'].iloc[-1])
        last_macd = float(df['MACD'].iloc[-1])
        last_signal = float(df['Signal_Line'].iloc[-1])

        # Your original action logic
        if last_rsi < 30 and last_macd > last_signal:
            action = "BUY"
        elif last_rsi > 70 and last_macd < last_signal:
            action = "SELL"
        else:
            action = "HOLD"

        # Your original news/sentiment
        news_list = fetch_news(ticker)
        sentiment_score = analyze_sentiment(news_list)
        sentiment = generate_signal(news_list, pred, sentiment_score)

        # ORIGINAL RESPONSE (maintained for compatibility)
        basic_response = {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "model_used": model_used,
            "prediction_price": round(pred, 4),
            "last_close": round(last_close, 4),
            "RSI": round(last_rsi, 2),
            "MACD": round(last_macd, 4),
            "Signal_line": round(last_signal, 4),
            "suggested_action": action,
            "news_sentiment": sentiment,
            "news": news_list[:5],
        }

        # ENHANCED ANALYSIS (new features)
        enhanced_analysis = None
        explanation_data = None

        try:
            # Get enhanced quote data
            quote = market_service.get_stock_quote(ticker)
            if quote:
                # Enhanced prediction with new ML models
                historical_data = market_service.get_historical_data(ticker, "1y", "1d")
                df_enhanced = market_service._dict_to_df(historical_data) if not historical_data.get("empty") else None
                
                if df_enhanced is not None and not df_enhanced.empty:
                    prediction_result = prediction_service.predict_stock_price(df_enhanced, ticker, horizon_days, model)
                    
                    if prediction_result:
                        # Generate trading signal
                        signal_result = prediction_service.generate_trading_signal(df_enhanced, ticker, prediction_result)
                        
                        enhanced_analysis = {
                            "advanced_prediction": {
                                "predicted_price": prediction_result.predicted_price,
                                "confidence_score": prediction_result.confidence_score,
                                "model_accuracy": prediction_result.model_accuracy,
                                "volatility_score": prediction_result.volatility_score,
                                "trend_direction": prediction_result.trend_direction.value,
                                "support_levels": prediction_result.support_levels,
                                "resistance_levels": prediction_result.resistance_levels
                            },
                            "trading_signal": {
                                "signal": signal_result.signal.value if signal_result else "HOLD",
                                "confidence": signal_result.confidence if signal_result else 0.5,
                                "target_price": signal_result.target_price if signal_result else None,
                                "stop_loss": signal_result.stop_loss if signal_result else None,
                                "risk_level": signal_result.risk_level if signal_result else "medium",
                                "reasoning": signal_result.reasoning if signal_result else []
                            },
                            "technical_indicators": {
                                "rsi": prediction_result.technical_indicators.rsi if prediction_result.technical_indicators else last_rsi,
                                "macd": prediction_result.technical_indicators.macd if prediction_result.technical_indicators else last_macd,
                                "macd_signal": prediction_result.technical_indicators.macd_signal if prediction_result.technical_indicators else last_signal,
                                "sma_20": prediction_result.technical_indicators.sma_20 if prediction_result.technical_indicators else None,
                                "sma_50": prediction_result.technical_indicators.sma_50 if prediction_result.technical_indicators else None,
                                "bb_upper": prediction_result.technical_indicators.bb_upper if prediction_result.technical_indicators else None,
                                "bb_lower": prediction_result.technical_indicators.bb_lower if prediction_result.technical_indicators else None,
                                "atr": prediction_result.technical_indicators.atr if prediction_result.technical_indicators else None
                            }
                        }

                        # AI EXPLANATION (new feature)
                        if include_explanation:
                            try:
                                context = ExplanationContext(
                                    user_experience_level=ComplexityLevel(complexity),
                                    preferred_tone=ToneStyle(tone),
                                    include_educational=complexity == "beginner"
                                )
                                
                                explanation = await explanation_service.explain_stock_analysis(ticker, context)
                                
                                if explanation:
                                    explanation_data = {
                                        "title": explanation.title,
                                        "summary": explanation.summary,
                                        "key_insights": [
                                            {
                                                "category": insight.category,
                                                "title": insight.title,
                                                "description": insight.description,
                                                "importance": insight.importance,
                                                "confidence": insight.confidence
                                            }
                                            for insight in explanation.key_insights[:5]
                                        ],
                                        "recommendations": explanation.recommendations,
                                        "risk_warnings": explanation.risk_warnings,
                                        "educational_notes": explanation.educational_notes if complexity == "beginner" else [],
                                        "confidence_score": explanation.confidence_score,
                                        "data_sources": explanation.data_sources
                                    }
                            except Exception as e:
                                print(f"Explanation generation failed: {e}")

                        # Enhanced news sentiment
                        if include_news:
                            try:
                                sentiment_analysis = await news_service.analyze_sentiment(ticker, hours_back=24)
                                if sentiment_analysis:
                                    enhanced_analysis["news_sentiment"] = {
                                        "overall_sentiment": sentiment_analysis.overall_sentiment.value,
                                        "sentiment_score": sentiment_analysis.sentiment_score,
                                        "confidence": sentiment_analysis.confidence,
                                        "total_articles": sentiment_analysis.total_articles,
                                        "trending_keywords": sentiment_analysis.trending_keywords[:5],
                                        "sentiment_trend": sentiment_analysis.sentiment_trend,
                                        "breaking_news_count": len(sentiment_analysis.recent_breaking_news)
                                    }
                            except Exception as e:
                                print(f"Enhanced news sentiment failed: {e}")

        except Exception as e:
            print(f"Enhanced analysis failed: {e}")

        # DATABASE LOGGING (your existing pattern enhanced)
        if db:
            try:
                # Enhanced logging with more data
                log_data = {
                    "RSI": last_rsi, 
                    "MACD": last_macd, 
                    "Signal": last_signal
                }
                
                if enhanced_analysis:
                    log_data.update({
                        "enhanced_prediction": enhanced_analysis.get("advanced_prediction", {}).get("predicted_price"),
                        "confidence": enhanced_analysis.get("advanced_prediction", {}).get("confidence_score"),
                        "volatility": enhanced_analysis.get("advanced_prediction", {}).get("volatility_score")
                    })

                log = AnalysisLog(
                    ticker=ticker,
                    model_used=model_used,
                    predicted=pred,
                    action=action,
                    indicators=log_data,
                    sentiment=sentiment
                )
                db.add(log)
                db.commit()
            except Exception as e:
                print(f"Database logging failed: {e}")

        # RESPONSE ASSEMBLY
        response = {
            "ticker": ticker.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_type": "comprehensive" if enhanced_analysis else "basic",
            
            # Original analysis (backward compatible)
            **basic_response,
            
            # Enhanced features
            "enhanced_analysis": enhanced_analysis,
            "ai_explanation": explanation_data,
            "confidence_score": enhanced_analysis.get("advanced_prediction", {}).get("confidence_score", 0.5) if enhanced_analysis else 0.5,
            "data_sources": explanation_data.get("data_sources", ["Market Data", "Technical Analysis"]) if explanation_data else ["Market Data", "Technical Analysis"]
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/{ticker}/comprehensive")
async def comprehensive_analysis(
    ticker: str,
    data_collector: DataCollector = Depends(get_data_collector_dep),
    explanation_service: ExplanationService = Depends(get_explanation_service_dep)
):
    """
    New comprehensive analysis endpoint using data collector
    """
    try:
        # Use data collector for comprehensive analysis
        analysis_data = await data_collector.collect_stock_analysis(ticker)
        
        if not analysis_data or "error" in analysis_data:
            raise HTTPException(status_code=404, detail="Could not perform comprehensive analysis")
        
        # Generate AI explanation
        context = ExplanationContext(
            user_experience_level=ComplexityLevel.INTERMEDIATE,
            preferred_tone=ToneStyle.CONVERSATIONAL
        )
        
        explanation = await explanation_service.explain_stock_analysis(ticker, context)
        
        return {
            "ticker": ticker.upper(),
            "timestamp": analysis_data.get("timestamp", datetime.utcnow().isoformat()),
            "comprehensive_data": analysis_data,
            "ai_explanation": {
                "summary": explanation.summary if explanation else "Analysis completed successfully",
                "key_insights": [asdict(insight) for insight in explanation.key_insights[:5]] if explanation else [],
                "recommendations": explanation.recommendations if explanation else [],
                "confidence": explanation.confidence_score if explanation else 0.5
            } if explanation else None,
            "execution_time": analysis_data.get("collection_time", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@router.post("/{ticker}/explain")
async def explain_analysis(
    ticker: str,
    request: AnalysisRequest,
    explanation_service: ExplanationService = Depends(get_explanation_service_dep)
):
    """
    Generate AI explanation for stock analysis
    """
    try:
        context = ExplanationContext(
            user_experience_level=ComplexityLevel(request.complexity_level),
            preferred_tone=ToneStyle(request.tone),
            include_educational=request.complexity_level == "beginner"
        )
        
        explanation = await explanation_service.explain_stock_analysis(ticker, context)
        
        if not explanation:
            raise HTTPException(status_code=404, detail="Could not generate explanation")
        
        return {
            "ticker": ticker.upper(),
            "explanation_type": explanation.explanation_type.value,
            "title": explanation.title,
            "summary": explanation.summary,
            "detailed_analysis": explanation.detailed_analysis,
            "key_insights": [
                {
                    "category": insight.category,
                    "title": insight.title,
                    "description": insight.description,
                    "importance": insight.importance,
                    "confidence": insight.confidence
                }
                for insight in explanation.key_insights
            ],
            "recommendations": explanation.recommendations,
            "risk_warnings": explanation.risk_warnings,
            "educational_notes": explanation.educational_notes,
            "confidence_score": explanation.confidence_score,
            "data_sources": explanation.data_sources,
            "methodology": explanation.methodology,
            "timestamp": explanation.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.post("/batch")
async def batch_analysis(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    data_collector: DataCollector = Depends(get_data_collector_dep)
):
    """
    Batch analysis for multiple tickers
    """
    try:
        if len(request.tickers) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 tickers allowed")
        
        if request.analysis_type == "comprehensive":
            # Use data collector for comprehensive batch analysis
            results = {}
            for ticker in request.tickers:
                try:
                    analysis = await data_collector.collect_stock_data(ticker)
                    results[ticker] = {
                        "status": "success",
                        "data": analysis,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    results[ticker] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            return {
                "analysis_type": request.analysis_type,
                "results": results,
                "summary": {
                    "total_requested": len(request.tickers),
                    "successful": len([r for r in results.values() if r["status"] == "success"]),
                    "failed": len([r for r in results.values() if r["status"] == "error"])
                }
            }
        else:
            # Basic batch analysis using original logic
            results = {}
            for ticker in request.tickers:
                try:
                    # Use your original analysis logic for basic batch
                    data_payload = fetch_historical_data(ticker)
                    df = to_df(data_payload)
                    
                    if not df.empty:
                        df = add_indicators(df)
                        pred, model_used = forecast(df, horizon_days=1, model="svr")
                        
                        results[ticker] = {
                            "status": "success",
                            "prediction_price": round(pred, 4),
                            "last_close": round(float(df['Close'].iloc[-1]), 4),
                            "RSI": round(float(df['RSI'].iloc[-1]), 2),
                            "model_used": model_used
                        }
                    else:
                        results[ticker] = {"status": "error", "error": "No data available"}
                        
                except Exception as e:
                    results[ticker] = {"status": "error", "error": str(e)}
            
            return {
                "analysis_type": "basic",
                "results": results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/{ticker}/signals")
async def get_trading_signals(
    ticker: str,
    timeframe: str = Query("1d", regex="^(1d|1h|4h)$"),
    prediction_service: PredictionService = Depends(get_prediction_service_dep),
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Get advanced trading signals for a ticker
    """
    try:
        # Get historical data
        historical_data = market_service.get_historical_data(ticker, "1y", timeframe)
        if historical_data.get("empty"):
            raise HTTPException(status_code=404, detail="No historical data available")
        
        df = market_service._dict_to_df(historical_data)
        
        # Generate prediction and signal
        prediction = prediction_service.predict_stock_price(df, ticker)
        if not prediction:
            raise HTTPException(status_code=500, detail="Could not generate prediction")
        
        signal = prediction_service.generate_trading_signal(df, ticker, prediction)
        if not signal:
            raise HTTPException(status_code=500, detail="Could not generate signal")
        
        return {
            "ticker": ticker.upper(),
            "timeframe": timeframe,
            "signal": {
                "action": signal.signal.value,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "risk_level": signal.risk_level,
                "hold_period": signal.hold_period
            },
            "prediction": {
                "current_price": prediction.current_price,
                "predicted_price": prediction.predicted_price,
                "price_change_percent": prediction.price_change_percent,
                "confidence": prediction.confidence_score,
                "model_used": prediction.model_used
            },
            "technical_levels": {
                "support": prediction.support_levels,
                "resistance": prediction.resistance_levels,
                "trend": prediction.trend_direction.value,
                "volatility": prediction.volatility_score
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

# LEGACY COMPATIBILITY ENDPOINT
@router.get("/legacy/{ticker}")
def analyze_stock_legacy(
    ticker: str,
    horizon_days: int = 1,
    model: str = "svr",
    db: Session = Depends(get_db)
):
    """
    Legacy endpoint - exact replica of your original analyze_stock function
    Maintained for backward compatibility
    """
    data_payload = fetch_historical_data(ticker)
    df = to_df(data_payload)
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data found")

    df = add_indicators(df)

    pred, model_used = forecast(df, horizon_days=horizon_days, model=model)

    last_close = float(df['Close'].iloc[-1])
    last_rsi = float(df['RSI'].iloc[-1])
    last_macd = float(df['MACD'].iloc[-1])
    last_signal = float(df['Signal_Line'].iloc[-1])

    if last_rsi < 30 and last_macd > last_signal:
        action = "BUY"
    elif last_rsi > 70 and last_macd < last_signal:
        action = "SELL"  # Fixed typo from your original
    else:
        action = "HOLD"  # Fixed typo from your original

    news_list = fetch_news(ticker)
    sentiment_score = analyze_sentiment(news_list)
    sentiment = generate_signal(news_list, pred, sentiment_score)

    resp = {
        "ticker": ticker,
        "horizon_days": horizon_days,
        "model_used": model_used,
        "prediction_price": round(pred, 4),
        "last_close": round(last_close, 4),
        "RSI": round(last_rsi, 2),
        "MACD": round(last_macd, 4),
        "Signal_line": round(last_signal, 4),
        "suggested_action": action,
        "news_sentiment": sentiment,
        "news": news_list[:5],
    }

    if db:
        log = AnalysisLog(
            ticker=ticker,
            model_used=model_used,
            predicted=pred,
            action=action,
            indicators={"RSI": last_rsi, "MACD": last_macd, "Signal": last_signal},
            sentiment=sentiment
        )
        db.add(log)
        db.commit()

    return resp

