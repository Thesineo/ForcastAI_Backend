from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.market_data import fetch_historical_data
from app.services.market_data import to_df
from app.models.svm_models import predict_stock_svr
from app.services.indicators import add_indicators
from app.services.forecasting import forecast
from app.services.news import fetch_news
from app.services.signals import generate_signal
from app.db.db import SessionLocal
from app.db.model import AnalysisLog
from app.core.config import settings
from app.services.sentiment import analyze_sentiment






router = APIRouter()


def get_db():
   if SessionLocal:
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()
   else:
       yield None






@router.get("/analyze/{ticker}")
def analyze_stock(
   ticker: str,
   horizon_days: int = 1,
   model: str = "svr",
   db: Session = Depends(get_db)


):
   data_payloadn = fetch_historical_data(ticker)
   df = to_df(data_payloadn)
   if df.empty:
       raise HTTPException(status_code=404, detail="No historical data found")
  
   df = add_indicators(df)


   pred, model_used = forecast(df, horizon_days=horizon_days, model=model)


   last_close = float(df['Close'].iloc[-1])
   last_rsi = float(df['RSI'].iloc[-1])
   last_macd = float(df['MACD'].iloc[-1])
   last_signal = float(df['Signal_Line'].iloc[-1])




   if last_rsi<30 and last_macd >last_signal:
       action = "BUY"
   elif last_rsi > 70 and last_macd< last_signal:
       actiion = "SELL"
   else:
       action = "Hold"


   news_list = fetch_news(ticker)
   sentiment_score =analyze_sentiment(news_list)
   sentiment = generate_signal(news_list, pred, sentiment_score)


   resp = {
       "ticker": ticker,
       "horizon_days": horizon_days,
       "model_used": model_used,
       "prediction_price": round(pred,4),
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


