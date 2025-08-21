import pandas as pd
import numpy as np


try:
   from prophet import Prophet
   HAS_PROPHET = True
except Exception:
   HAS_PROPHET = False




from app.models.svm_models import predict_stock_svr


def forecast_with_prophet(df: pd.DataFrame, horizon_days: int = 1) -> float:
   if not HAS_PROPHET:
       raise RuntimeError("Prophet not available")
   pdf = df[['Close']].reset_index()
   pdf.columns = ['ds', 'y']
   m = Prophet(daily_seasonality=True)
   m.fit(pdf)
   future = m.make_future_dataframe(periods=horizon_days)
   forecast = m.predict(future)
   return float(forecast['yhat'].iloc[-1])




def forecast(df: pd.DataFrame, horizon_days: int = 1, model: str = "svr") -> tuple[float, str]:
   """
   Returns (prediction, model_used)
   model: "svr" | "prophet" | "xgb"
   """
   if model == "prophet" and HAS_PROPHET:
       return forecast_with_prophet(df, horizon_days), "prophet"
  
   else:
       # default fallback to SVR
       return predict_stock_svr(df, horizon=horizon_days), "svr"








