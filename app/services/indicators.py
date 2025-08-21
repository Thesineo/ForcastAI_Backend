import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14):
   delta = series.diff()
   gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
   loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
   rs = gain / (loss + 1e-10)
   return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
   ema_fast = series.ewm(span=fast, adjust=False).mean()
   ema_slow = series.ewm(span=slow, adjust=False).mean()
   macd_line = ema_fast - ema_slow
   signal_line = macd_line.ewm(span=signal, adjust=False).mean()
   hist = macd_line - signal_line
   return macd_line, signal_line, hist


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()
   df['SMA_20'] = df['Close'].rolling(window=20).mean()
   df['SMA_50'] = df['Close'].rolling(window=50).mean()
   df['RSI'] = rsi(df['Close'])
   macd_line, signal_line, hist = macd(df['Close'])
   df['MACD'] = macd_line
   df['Signal_Line'] = signal_line
   df['MACD_Hist'] = hist
   return df.dropna()



