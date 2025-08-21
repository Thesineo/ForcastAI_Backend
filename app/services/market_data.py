import yfinance as yf
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from app.services.cache import cache_result


@cache_result(ttl=3600, key_prefix="market_" )
def fetch_historical_data(ticker: str, period: str = "5y", interval: str = "1d") -> dict:
   df = yf.download(ticker, period=period, interval=interval, progress=False)
   if df.empty:
       return {"empty": True}
   df = df.dropna()
   return df.reset_index().to_dict(orient="list")


def to_df(payload: dict) -> pd.DataFrame:
   if not payload or "empty" in payload:
       return pd.DataFrame()
   return pd.DataFrame(payload).set_index("Date")




