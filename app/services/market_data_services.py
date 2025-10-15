"""
Market Data Service for AI Financial Advisor
Handles real-time and historical market data retrieval, processing, and caching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import our enhanced cache system
from .cache import CacheService, CacheType, get_cache_service, cached 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataInterval(Enum):
    """Data interval enumeration"""
    MINUTE_1 = "1m"
    MINUTE_2 = "2m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_60 = "60m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"
    MONTH_3 = "3mo"

class DataPeriod(Enum):
    """Data period enumeration"""
    DAY_1 = "1d"
    DAY_5 = "5d"
    MONTH_1 = "1mo"
    MONTH_3 = "3mo"
    MONTH_6 = "6mo"
    YEAR_1 = "1y"
    YEAR_2 = "2y"
    YEAR_5 = "5y"
    YEAR_10 = "10y"
    YTD = "ytd"
    MAX = "max"

@dataclass
class StockQuote:
    """Stock quote data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class MarketSummary:
    """Market summary data structure"""
    indices: Dict[str, Dict] = None
    top_gainers: List[Dict] = None
    top_losers: List[Dict] = None
    most_active: List[Dict] = None
    market_status: str = "unknown"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class CompanyInfo:
    """Company information data structure"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class MarketDataService:
    """Comprehensive market data service"""
    
    # Major market indices
    MAJOR_INDICES = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX"
    }
    
    def __init__(self, cache_service: CacheService = None):
        self.cache = cache_service or get_cache_service()
        self.session = self._create_session()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame"""
        if df.empty:
            return df
            
        # Remove NaN values
        df = df.dropna()
        
        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Round numerical values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        return df
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to dictionary format for caching/API response"""
        if df.empty:
            return {"empty": True, "error": "No data available"}
            
        df_clean = self._clean_dataframe(df)
        result = df_clean.reset_index().to_dict(orient="records")
        
        return {
            "data": result,
            "columns": list(df_clean.columns),
            "index_name": df_clean.index.name or "Date",
            "shape": df_clean.shape,
            "empty": False
        }
    
    def _dict_to_df(self, data_dict: Dict) -> pd.DataFrame:
        """Convert dictionary back to DataFrame"""
        if not data_dict or data_dict.get("empty", True):
            return pd.DataFrame()
            
        df = pd.DataFrame(data_dict["data"])
        index_name = data_dict.get("index_name", "Date")
        
        if index_name in df.columns:
            df = df.set_index(index_name)
            df.index = pd.to_datetime(df.index)
            
        return df
    
    @cached(CacheType.STOCK_PRICES, ttl=60)
    def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time stock quote"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if hist.empty or not info:
                logger.warning(f"No data found for symbol: {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
            
            quote = StockQuote(
                symbol=symbol.upper(),
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('forwardPE'),
                dividend_yield=info.get('dividendYield'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
                avg_volume=info.get('averageVolume')
            )
            
            logger.info(f"Retrieved quote for {symbol}: ${current_price:.2f}")
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    @cached(CacheType.MARKET_DATA, ttl=300)
    def get_historical_data(self, 
                          symbol: str, 
                          period: Union[str, DataPeriod] = DataPeriod.YEAR_1,
                          interval: Union[str, DataInterval] = DataInterval.DAY_1) -> Dict:
        """Get historical price data"""
        try:
            # Convert enums to string values
            if isinstance(period, DataPeriod):
                period = period.value
            if isinstance(interval, DataInterval):
                interval = interval.value
                
            # Validate interval vs period combinations
            if not self._validate_period_interval(period, interval):
                raise ValueError(f"Invalid period-interval combination: {period}-{interval}")
            
            ticker = yf.Ticker(symbol.upper())
            df = ticker.history(period=period, interval=interval, progress=False)
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return {"empty": True, "error": f"No data found for {symbol}"}
            
            # Add technical indicators
            df = self._add_basic_indicators(df)
            
            result = self._df_to_dict(df)
            result["symbol"] = symbol.upper()
            result["period"] = period
            result["interval"] = interval
            
            logger.info(f"Retrieved {len(df)} records for {symbol} ({period}, {interval})")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {"empty": True, "error": str(e)}
    
    def _validate_period_interval(self, period: str, interval: str) -> bool:
        """Validate period and interval combinations"""
        # yfinance has restrictions on certain combinations
        intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m"]
        long_periods = ["2y", "5y", "10y", "max"]
        
        if interval in intraday_intervals and period in long_periods:
            return False
            
        return True
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to price data"""
        if df.empty:
            return df
            
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    @cached(CacheType.MARKET_DATA, ttl=900)  # 15 minutes
    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """Get comprehensive company information"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No company info found for {symbol}")
                return None
            
            company_info = CompanyInfo(
                symbol=symbol.upper(),
                name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                enterprise_value=info.get('enterpriseValue'),
                pe_ratio=info.get('forwardPE') or info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                roe=info.get('returnOnEquity'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
                description=info.get('longBusinessSummary'),
                website=info.get('website'),
                employees=info.get('fullTimeEmployees')
            )
            
            logger.info(f"Retrieved company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return None
    
    @cached(CacheType.MARKET_DATA, ttl=300)  # 5 minutes
    def get_market_summary(self) -> Optional[MarketSummary]:
        """Get market summary with major indices and movers"""
        try:
            indices = {}
            
            # Get major indices
            for name, symbol in self.MAJOR_INDICES.items():
                quote = self.get_stock_quote(symbol)
                if quote:
                    indices[name] = {
                        'symbol': symbol,
                        'price': quote.price,
                        'change': quote.change,
                        'change_percent': quote.change_percent
                    }
            
            # Get market movers (simplified version)
            # In a production app, you'd want to use a proper API for this
            movers = self._get_market_movers()
            
            summary = MarketSummary(
                indices=indices,
                top_gainers=movers.get('gainers', []),
                top_losers=movers.get('losers', []),
                most_active=movers.get('active', []),
                market_status=self._get_market_status()
            )
            
            logger.info("Retrieved market summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error fetching market summary: {e}")
            return None
    
    def _get_market_movers(self) -> Dict[str, List]:
        """Get top market movers (simplified implementation)"""
        # This is a simplified implementation
        # In production, you'd use a proper API like Alpha Vantage, Polygon, etc.
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        movers = {'gainers': [], 'losers': [], 'active': []}
        
        for symbol in popular_stocks:
            quote = self.get_stock_quote(symbol)
            if quote:
                stock_data = {
                    'symbol': quote.symbol,
                    'price': quote.price,
                    'change': quote.change,
                    'change_percent': quote.change_percent,
                    'volume': quote.volume
                }
                
                if quote.change_percent > 2:
                    movers['gainers'].append(stock_data)
                elif quote.change_percent < -2:
                    movers['losers'].append(stock_data)
                    
                if quote.volume > 1000000:  # High volume threshold
                    movers['active'].append(stock_data)
        
        # Sort by change percent
        movers['gainers'] = sorted(movers['gainers'], key=lambda x: x['change_percent'], reverse=True)[:5]
        movers['losers'] = sorted(movers['losers'], key=lambda x: x['change_percent'])[:5]
        movers['active'] = sorted(movers['active'], key=lambda x: x['volume'], reverse=True)[:5]
        
        return movers
    
    def _get_market_status(self) -> str:
        """Determine current market status"""
        now = datetime.now(timezone.utc)
        
        # Convert to Eastern Time (market timezone)
        eastern = now.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
        
        # Check if it's a weekday
        if eastern.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return "closed"
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open <= eastern <= market_close:
            return "open"
        elif eastern < market_open:
            return "pre_market"
        else:
            return "after_hours"
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Optional[StockQuote]]:
        """Get quotes for multiple symbols concurrently"""
        try:
            # Use ThreadPoolExecutor for concurrent requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(self.get_stock_quote, symbol): symbol 
                    for symbol in symbols
                }
                
                results = {}
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        results[symbol] = future.result(timeout=30)
                    except Exception as e:
                        logger.error(f"Error fetching quote for {symbol}: {e}")
                        results[symbol] = None
                
                return results
                
        except Exception as e:
            logger.error(f"Error fetching multiple quotes: {e}")
            return {symbol: None for symbol in symbols}
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for stocks by name or symbol"""
        try:
            # This is a basic implementation using yfinance
            # For production, consider using a proper search API
            
            # Try direct ticker lookup first
            try:
                ticker = yf.Ticker(query.upper())
                info = ticker.info
                if info and 'symbol' in info:
                    return [{
                        'symbol': info.get('symbol', query.upper()),
                        'name': info.get('longName', info.get('shortName', 'Unknown')),
                        'type': 'stock',
                        'exchange': info.get('exchange', 'Unknown')
                    }]
            except:
                pass
            
            # For a more comprehensive search, you'd integrate with:
            # - Alpha Vantage Symbol Search
            # - Polygon.io Symbol Search
            # - IEX Cloud Symbol Search
            
            logger.info(f"Stock search performed for: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Error searching stocks for '{query}': {e}")
            return []
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get sector performance data"""
        try:
            # Sector ETFs as proxies for sector performance
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            }
            
            sector_performance = {}
            for sector, etf in sector_etfs.items():
                quote = self.get_stock_quote(etf)
                if quote:
                    sector_performance[sector] = {
                        'symbol': etf,
                        'price': quote.price,
                        'change': quote.change,
                        'change_percent': quote.change_percent
                    }
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
    
    def get_trending_stocks(self, limit: int = 20) -> List[Dict]:
        """Get trending stocks (basic implementation)"""
        try:
            # This is a simplified implementation
            # In production, you'd use APIs like Yahoo Finance Trending, Reddit sentiment, etc.
            
            popular_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'AMD', 'CRM', 'ADBE', 'INTC', 'PYPL', 'UBER', 'ZOOM', 'SHOP',
                'SQ', 'ROKU', 'TWLO', 'DOCU'
            ]
            
            quotes = self.get_multiple_quotes(popular_tickers[:limit])
            
            trending = []
            for symbol, quote in quotes.items():
                if quote:
                    trending.append({
                        'symbol': quote.symbol,
                        'price': quote.price,
                        'change': quote.change,
                        'change_percent': quote.change_percent,
                        'volume': quote.volume
                    })
            
            # Sort by volume and change percentage
            trending.sort(key=lambda x: (x['volume'] * abs(x['change_percent'])), reverse=True)
            
            return trending[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching trending stocks: {e}")
            return []
    
    def get_earnings_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming earnings (basic implementation)"""
        try:
            # This would require a specialized API in production
            # For now, return empty list with proper structure
            logger.info(f"Earnings calendar requested for next {days_ahead} days")
            
            # In production, integrate with:
            # - Alpha Vantage Earnings Calendar
            # - Polygon.io Earnings Calendar
            # - Yahoo Finance Earnings Calendar
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform market data service health check"""
        try:
            start_time = time.time()
            
            # Test basic functionality
            test_quote = self.get_stock_quote("AAPL")
            
            response_time = time.time() - start_time
            
            return {
                "service": "market_data",
                "status": "healthy" if test_quote else "unhealthy",
                "response_time": round(response_time, 3),
                "test_symbol": "AAPL",
                "test_successful": bool(test_quote),
                "cache_stats": self.cache.get_stats() if self.cache else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "service": "market_data",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)

# Utility functions for backward compatibility
def fetch_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> Dict:
    """Legacy function for backward compatibility"""
    service = MarketDataService()
    return service.get_historical_data(ticker, period, interval)

def to_df(payload: Dict) -> pd.DataFrame:
    """Convert payload back to DataFrame"""
    service = MarketDataService()
    return service._dict_to_df(payload)

# Global service instance
_market_service = None

def get_market_service() -> MarketDataService:
    """Get global market data service instance"""
    global _market_service
    if _market_service is None:
        _market_service = MarketDataService()
    return _market_service

if __name__ == "__main__":
    # Example usage and testing
    service = MarketDataService()
    
    print("Testing Market Data Service...")
    
    # Test stock quote
    quote = service.get_stock_quote("AAPL")
    if quote:
        print(f"AAPL Quote: ${quote.price:.2f} ({quote.change_percent:+.2f}%)")
    
    # Test historical data
    historical = service.get_historical_data("AAPL", "1mo", "1d")
    if not historical.get("empty"):
        print(f"Historical data retrieved: {historical['shape']} records")
    
    # Test market summary
    summary = service.get_market_summary()
    if summary:
        print(f"Market Status: {summary.market_status}")
        print(f"Indices: {len(summary.indices)} loaded")
    
    # Health check
    health = service.health_check()
    print(f"Health Check: {health['status']}")
    
    # Clean up
    service.close()