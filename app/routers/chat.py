# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.market_data import fetch_historical_data, to_df
from app.services.indicators import add_indicators

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_with_ai(req: ChatRequest):
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
