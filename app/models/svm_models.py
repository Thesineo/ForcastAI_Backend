import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _make_features(df: pd.DataFrame, lookback: int = 20):
   """
   Build simple lag-based features for SVR.
   Each row contains `lookback` number of past prices, and the  label is the next day's price.
   """
   X, y = [], []
   prices = df['Close'].values.flatten()


   for i in range(lookback, len(prices)):
       window = prices[i - lookback:i].flatten()
       if len(window) == lookback:
           X.append(window)
           y.append(prices[i])
   X= np.array(X, dtype= float)
   y= np.array(y, dtype=float)
   return X, y 
       


def predict_stock_svr(df: pd.DataFrame, horizon: int = 1, lookback: int = 20) -> float:
   """
   Predict the next `horizon` days's stock price using SVR.
   Uses lag-based features (previous 'lookbook' days).
   """
   
   df = df.dropna().reset_index(drop=True) # Remove NaNs and reset index

   if len(df) < lookback + 2:
       raise ValueError("Not enough data to train SVR model.")
   
   X, y = _make_features(df, lookback=lookback)
   
  
   

   #Ensure x is 2D
   if X.ndim != 2:
      raise ValueError(f"Input features X must be 2D, got shape {X.shape}")
   
   print(f"Training data  shape: X={X.shape}, y={y.shape}")
   print(f"Horizon: {horizon}, Lookback: {lookback}")
   
   model= Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])
   model.fit(X, y)   
    
   

   # roll forward predictions (recursive)
   last_prices = df['Close'].values[-lookback:]
   window = []
   for price in last_prices:
       if isinstance(price, (list, np.ndarray)):
           window.append(float(price.flatten()[0]))
       else:
           window.append(float(price))

  
   print(f"Intial window length: {len(window)}")
   print(f"Intial window: {window}")

   pred = None
   for step in range(horizon):
       current_window = window[-lookback:]

       if len(current_window) != lookback:
          raise ValueError(f"Window size mismatch: expected {lookback}, got {len(current_window)}")
        
       clean_window = []
       for val in current_window:
           if isinstance(val, (list, np.ndarray)):
               clean_window.append(float(val.flatten()[0]))
           else:
               clean_window.append(float(val))

       print(f"Step {step+1}: clean_window length= {len(clean_window)}")
       print(f"Step {step+1}: clean_window types = {[type(x) for x in clean_window[:3]]}")

       try:
           x_next = np.array( current_window, dtype=float).reshape(1, -1)
           print(f"Step {step+1}: x_next shape = {x_next.shape}")
       except Exception as e:
           print(f"Error creating x_next: {e}")
           print(f"clean_window content: {clean_window}")
           raise
               
      
       
       pred = float(model.predict(x_next)[0])
       print(f"Step {step+1}: prediction = {pred}")
       window.append(float(pred))

       if len(window) > lookback * 2:
           window = window[-lookback-5:]

   return float(pred)