import pandas as pd

def generate_signal(data, forecast, sentiment_score):
    """Basic rule-based signal"""
    
    # Debug: Print data type and structure
    print(f"Data type: {type(data)}")
    print(f"Data keys/columns: {data.keys() if hasattr(data, 'keys') else 'No keys available'}")
    
    try:
        # Handle different data formats
        if isinstance(data, dict):
            # If data is a dictionary, convert to DataFrame
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            # If data is a list of dictionaries
            df = pd.DataFrame(data)
        else:
            # Assume it's already a DataFrame
            df = data
        
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Last few rows:\n{df.tail()}")
        
        # Try to find the close price column (case-insensitive)
        close_column = None
        for col in df.columns:
            if col.lower() in ['close', 'close_price', 'closing_price', 'adj_close']:
                close_column = col
                break
        
        if close_column is None:
            # If no close column found, try to use the first numeric column
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                close_column = numeric_columns[0]
                print(f"Using {close_column} as price column")
            else:
                print("No numeric columns found")
                return "HOLD"
        
        # Get the last price
        last_price = df[close_column].iloc[-1]
        print(f"Last price ({close_column}): {last_price}")
        print(f"Forecast: {forecast}")
        print(f"Sentiment score: {sentiment_score}")
        
        # Generate signal based on forecast vs last price and sentiment
        if forecast > last_price and sentiment_score > 0.3:
            return "BUY"
        elif sentiment_score < 0:
            return "SELL"
        else:
            return "HOLD"
            
    except Exception as e:
        print(f"Error in generate_signal: {str(e)}")
        print(f"Data received: {data}")
        return "HOLD"
   


