"""
Data Collector Service for AI Financial Advisor
Orchestrates and coordinates data collection from multiple sources
Provides unified interface for frontend data consumption
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from functools import wraps
import json

# Import our services
from app.services.cache import CacheService, CacheType, get_cache_service
from app.services.market_data_services import MarketDataService, get_market_service, StockQuote, MarketSummary



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source enumeration"""
    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL_SENTIMENT = "social_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    INSIDER_TRADING = "insider_trading"
    ANALYST_RATINGS = "analyst_ratings"
    EARNINGS = "earnings"
    DIVIDENDS = "dividends"

class Priority(Enum):
    """Data collection priority levels"""
    HIGH = 1      # Real-time critical data (prices, alerts)
    MEDIUM = 2    # Important but can wait (news, sentiment)
    LOW = 3       # Background data (historical, company info)

@dataclass
class DataRequest:
    """Data collection request structure"""
    source: DataSource
    symbol: str
    data_type: str
    parameters: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    callback: Optional[callable] = None
    timeout: int = 30
    retry_count: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class CollectionResult:
    """Data collection result structure"""
    request_id: str
    source: DataSource
    symbol: str
    data: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class PortfolioData:
    """Portfolio data aggregation structure"""
    user_id: str
    holdings: List[Dict]
    total_value: float
    day_change: float
    day_change_percent: float
    positions: Dict[str, Dict]
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class DashboardData:
    """Complete dashboard data structure"""
    market_summary: Optional[MarketSummary]
    portfolio: Optional[PortfolioData]
    watchlist: List[StockQuote]
    trending_stocks: List[Dict]
    news_headlines: List[Dict]
    market_alerts: List[Dict]
    performance_summary: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, cache_service: CacheService = None, market_service: MarketDataService = None):
        self.cache = cache_service or get_cache_service()
        self.market_service = market_service or get_market_service()
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.active_collections = {}
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        self._lock = threading.Lock()
        
    async def collect_stock_data(self, 
                               symbol: str, 
                               include_historical: bool = True,
                               include_company_info: bool = False,
                               historical_period: str = "1y") -> Dict[str, Any]:
        """Collect comprehensive stock data"""
        start_time = time.time()
        
        try:
            # Parallel data collection
            tasks = []
            
            # Real-time quote (high priority)
            tasks.append(self._collect_quote_async(symbol))
            
            # Historical data if requested
            if include_historical:
                tasks.append(self._collect_historical_async(symbol, historical_period))
            
            # Company info if requested
            if include_company_info:
                tasks.append(self._collect_company_info_async(symbol))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            stock_data = {
                'symbol': symbol.upper(),
                'quote': None,
                'historical': None,
                'company_info': None,
                'collection_time': time.time() - start_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Map results based on task order
            if len(results) > 0 and not isinstance(results[0], Exception):
                stock_data['quote'] = results[0]
            
            if include_historical and len(results) > 1 and not isinstance(results[1], Exception):
                stock_data['historical'] = results[1]
            
            if include_company_info and len(results) > 2 and not isinstance(results[2], Exception):
                stock_data['company_info'] = results[2]
            
            logger.info(f"Collected stock data for {symbol} in {stock_data['collection_time']:.2f}s")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error collecting stock data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _collect_quote_async(self, symbol: str) -> Optional[Dict]:
        """Async wrapper for stock quote collection"""
        loop = asyncio.get_event_loop()
        try:
            quote = await loop.run_in_executor(
                self.executor, 
                self.market_service.get_stock_quote, 
                symbol
            )
            return asdict(quote) if quote else None
        except Exception as e:
            logger.error(f"Error collecting quote for {symbol}: {e}")
            return None
    
    async def _collect_historical_async(self, symbol: str, period: str) -> Optional[Dict]:
        """Async wrapper for historical data collection"""
        loop = asyncio.get_event_loop()
        try:
            historical = await loop.run_in_executor(
                self.executor,
                self.market_service.get_historical_data,
                symbol, period, "1d"
            )
            return historical if not historical.get('empty') else None
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return None
    
    async def _collect_company_info_async(self, symbol: str) -> Optional[Dict]:
        """Async wrapper for company info collection"""
        loop = asyncio.get_event_loop()
        try:
            company_info = await loop.run_in_executor(
                self.executor,
                self.market_service.get_company_info,
                symbol
            )
            return asdict(company_info) if company_info else None
        except Exception as e:
            logger.error(f"Error collecting company info for {symbol}: {e}")
            return None
    
    async def collect_portfolio_data(self, 
                                   user_id: str, 
                                   holdings: List[Dict]) -> Optional[PortfolioData]:
        """Collect and aggregate portfolio data"""
        start_time = time.time()
        
        try:
            if not holdings:
                return None
            
            # Extract symbols from holdings
            symbols = [holding['symbol'] for holding in holdings]
            
            # Get current quotes for all holdings
            quotes_dict = await self._collect_multiple_quotes_async(symbols)
            
            # Calculate portfolio metrics
            total_value = 0.0
            day_change = 0.0
            positions = {}
            
            for holding in holdings:
                symbol = holding['symbol']
                quantity = holding.get('quantity', 0)
                avg_cost = holding.get('avg_cost', 0)
                
                quote = quotes_dict.get(symbol)
                if quote:
                    current_price = quote['price']
                    position_value = quantity * current_price
                    position_gain_loss = quantity * (current_price - avg_cost)
                    position_change = quantity * quote['change']
                    
                    total_value += position_value
                    day_change += position_change
                    
                    positions[symbol] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'avg_cost': avg_cost,
                        'current_price': current_price,
                        'position_value': position_value,
                        'unrealized_pnl': position_gain_loss,
                        'day_change': position_change,
                        'day_change_percent': (position_change / (position_value - position_change)) * 100 if position_value != position_change else 0,
                        'weight': 0  # Will calculate after total_value is known
                    }
            
            # Calculate position weights
            for position in positions.values():
                position['weight'] = (position['position_value'] / total_value) * 100 if total_value > 0 else 0
            
            day_change_percent = (day_change / (total_value - day_change)) * 100 if total_value != day_change else 0
            
            # Calculate performance metrics
            performance = self._calculate_portfolio_performance(positions)
            
            # Calculate risk metrics  
            risk_metrics = self._calculate_risk_metrics(positions, quotes_dict)
            
            portfolio_data = PortfolioData(
                user_id=user_id,
                holdings=holdings,
                total_value=total_value,
                day_change=day_change,
                day_change_percent=day_change_percent,
                positions=positions,
                performance=performance,
                risk_metrics=risk_metrics
            )
            
            # Cache portfolio data
            self.cache.set(CacheType.PORTFOLIO, user_id, asdict(portfolio_data), ttl=300)
            
            logger.info(f"Collected portfolio data for {user_id} in {time.time() - start_time:.2f}s")
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error collecting portfolio data for {user_id}: {e}")
            return None
    
    async def _collect_multiple_quotes_async(self, symbols: List[str]) -> Dict[str, Dict]:
        """Async collection of multiple quotes"""
        loop = asyncio.get_event_loop()
        try:
            quotes_dict = await loop.run_in_executor(
                self.executor,
                self.market_service.get_multiple_quotes,
                symbols
            )
            
            # Convert StockQuote objects to dictionaries
            result = {}
            for symbol, quote in quotes_dict.items():
                result[symbol] = asdict(quote) if quote else None
                
            return result
        except Exception as e:
            logger.error(f"Error collecting multiple quotes: {e}")
            return {}
    
    def _calculate_portfolio_performance(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            total_value = sum(pos['position_value'] for pos in positions.values())
            total_cost = sum(pos['quantity'] * pos['avg_cost'] for pos in positions.values())
            
            if total_cost == 0:
                return {'total_return': 0.0, 'total_return_percent': 0.0}
            
            total_return = total_value - total_cost
            total_return_percent = (total_return / total_cost) * 100
            
            # Additional performance metrics
            winning_positions = sum(1 for pos in positions.values() if pos['unrealized_pnl'] > 0)
            total_positions = len(positions)
            win_rate = (winning_positions / total_positions) * 100 if total_positions > 0 else 0
            
            return {
                'total_return': total_return,
                'total_return_percent': total_return_percent,
                'win_rate': win_rate,
                'winning_positions': winning_positions,
                'total_positions': total_positions
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {'total_return': 0.0, 'total_return_percent': 0.0}
    
    def _calculate_risk_metrics(self, positions: Dict[str, Dict], quotes_dict: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate basic risk metrics"""
        try:
            if not positions:
                return {'concentration_risk': 0.0, 'volatility_score': 0.0}
            
            # Concentration risk (largest position weight)
            max_weight = max(pos['weight'] for pos in positions.values())
            
            # Simple volatility score based on day changes
            day_changes = [abs(pos['day_change_percent']) for pos in positions.values()]
            avg_volatility = sum(day_changes) / len(day_changes) if day_changes else 0
            
            return {
                'concentration_risk': max_weight,
                'volatility_score': avg_volatility,
                'diversification_score': 100 - max_weight  # Simple diversification measure
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'concentration_risk': 0.0, 'volatility_score': 0.0}
    
    async def collect_watchlist_data(self, symbols: List[str]) -> List[Dict]:
        """Collect data for watchlist symbols"""
        start_time = time.time()
        
        try:
            if not symbols:
                return []
            
            quotes_dict = await self._collect_multiple_quotes_async(symbols)
            
            watchlist_data = []
            for symbol in symbols:
                quote_data = quotes_dict.get(symbol)
                if quote_data:
                    watchlist_data.append(quote_data)
            
            # Sort by change percentage (most active first)
            watchlist_data.sort(key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            
            logger.info(f"Collected watchlist data for {len(symbols)} symbols in {time.time() - start_time:.2f}s")
            return watchlist_data
            
        except Exception as e:
            logger.error(f"Error collecting watchlist data: {e}")
            return []
    
    async def collect_dashboard_data(self, 
                                   user_id: str,
                                   portfolio_holdings: List[Dict] = None,
                                   watchlist_symbols: List[str] = None) -> Optional[DashboardData]:
        """Collect complete dashboard data"""
        start_time = time.time()
        
        try:
            # Parallel collection of all dashboard components
            tasks = [
                self._collect_market_summary_async(),
                self._collect_trending_stocks_async(),
            ]
            
            # Add portfolio collection if holdings provided
            if portfolio_holdings:
                tasks.append(self.collect_portfolio_data(user_id, portfolio_holdings))
            
            # Add watchlist collection if symbols provided
            if watchlist_symbols:
                tasks.append(self.collect_watchlist_data(watchlist_symbols))
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            dashboard_data = DashboardData(
                market_summary=results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
                portfolio=None,
                watchlist=[],
                trending_stocks=results[1] if len(results) > 1 and not isinstance(results[1], Exception) else [],
                news_headlines=[],  # Will be populated when news service is ready
                market_alerts=[],   # Will be populated when alert service is ready
                performance_summary={}
            )
            
            # Handle portfolio data
            result_index = 2
            if portfolio_holdings and len(results) > result_index and not isinstance(results[result_index], Exception):
                dashboard_data.portfolio = results[result_index]
                result_index += 1
            
            # Handle watchlist data
            if watchlist_symbols and len(results) > result_index and not isinstance(results[result_index], Exception):
                dashboard_data.watchlist = results[result_index]
            
            # Create performance summary
            dashboard_data.performance_summary = self._create_performance_summary(dashboard_data)
            
            # Cache dashboard data
            cache_key = f"{user_id}_dashboard"
            self.cache.set(CacheType.USER_SESSION, cache_key, asdict(dashboard_data), ttl=300)
            
            logger.info(f"Collected dashboard data for {user_id} in {time.time() - start_time:.2f}s")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error collecting dashboard data for {user_id}: {e}")
            return None
    
    async def _collect_market_summary_async(self) -> Optional[Dict]:
        """Async wrapper for market summary collection"""
        loop = asyncio.get_event_loop()
        try:
            summary = await loop.run_in_executor(
                self.executor,
                self.market_service.get_market_summary
            )
            return asdict(summary) if summary else None
        except Exception as e:
            logger.error(f"Error collecting market summary: {e}")
            return None
    
    async def _collect_trending_stocks_async(self) -> List[Dict]:
        """Async wrapper for trending stocks collection"""
        loop = asyncio.get_event_loop()
        try:
            trending = await loop.run_in_executor(
                self.executor,
                self.market_service.get_trending_stocks,
                10
            )
            return trending or []
        except Exception as e:
            logger.error(f"Error collecting trending stocks: {e}")
            return []
    
    def _create_performance_summary(self, dashboard_data: DashboardData) -> Dict[str, Any]:
        """Create performance summary for dashboard"""
        try:
            summary = {
                'market_status': dashboard_data.market_summary.market_status if dashboard_data.market_summary else 'unknown',
                'portfolio_performance': {},
                'market_performance': {},
                'alerts_count': len(dashboard_data.market_alerts)
            }
            
            # Portfolio performance
            if dashboard_data.portfolio:
                summary['portfolio_performance'] = {
                    'total_value': dashboard_data.portfolio.total_value,
                    'day_change': dashboard_data.portfolio.day_change,
                    'day_change_percent': dashboard_data.portfolio.day_change_percent,
                    'positions_count': len(dashboard_data.portfolio.positions)
                }
            
            # Market performance from indices
            if dashboard_data.market_summary and dashboard_data.market_summary.indices:
                summary['market_performance'] = {
                    name: {
                        'change_percent': data.get('change_percent', 0)
                    } for name, data in dashboard_data.market_summary.indices.items()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating performance summary: {e}")
            return {}
    
    async def collect_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Collect comprehensive stock analysis data"""
        start_time = time.time()
        
        try:
            # Collect comprehensive data
            stock_data = await self.collect_stock_data(
                symbol, 
                include_historical=True,
                include_company_info=True,
                historical_period="1y"
            )
            
            # Add analysis components (will be enhanced when other services are ready)
            analysis = {
                'symbol': symbol.upper(),
                'basic_data': stock_data,
                'technical_analysis': {},  # Will be populated by prediction service
                'fundamental_analysis': {},  # Will be populated by analysis service
                'news_sentiment': {},  # Will be populated by news service
                'ai_insights': {},  # Will be populated by explanation service
                'recommendation': 'hold',  # Default recommendation
                'confidence_score': 0.5,  # Default confidence
                'risk_assessment': 'medium',  # Default risk level
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_time': time.time() - start_time
            }
            
            # Cache analysis data
            self.cache.set(CacheType.ANALYSIS, symbol, analysis, ttl=1800)  # 30 minutes
            
            logger.info(f"Collected stock analysis for {symbol} in {analysis['collection_time']:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error collecting stock analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_cached_data(self, cache_type: CacheType, key: str, **kwargs) -> Optional[Any]:
        """Get cached data with fallback"""
        try:
            cached_data = self.cache.get(cache_type, key, **kwargs)
            if cached_data:
                self.collection_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {cache_type.value}:{key}")
            return cached_data
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    def invalidate_cache(self, cache_type: CacheType, key: str = None, **kwargs) -> bool:
        """Invalidate specific cache entries"""
        try:
            if key:
                return self.cache.delete(cache_type, key, **kwargs)
            else:
                return self.cache.clear_cache_type(cache_type) > 0
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    async def refresh_data(self, 
                         symbols: List[str], 
                         data_types: List[str] = None,
                         force_refresh: bool = False) -> Dict[str, Any]:
        """Refresh data for multiple symbols"""
        start_time = time.time()
        
        try:
            if not symbols:
                return {}
            
            # Default data types
            if not data_types:
                data_types = ['quote', 'historical']
            
            # Clear cache if force refresh
            if force_refresh:
                for symbol in symbols:
                    self.cache.delete(CacheType.STOCK_PRICES, symbol)
                    if 'historical' in data_types:
                        self.cache.delete(CacheType.MARKET_DATA, symbol)
            
            # Collect fresh data
            refresh_results = {}
            tasks = []
            
            for symbol in symbols:
                tasks.append(self.collect_stock_data(
                    symbol,
                    include_historical='historical' in data_types,
                    include_company_info='company_info' in data_types
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, symbol in enumerate(symbols):
                if i < len(results) and not isinstance(results[i], Exception):
                    refresh_results[symbol] = results[i]
                else:
                    refresh_results[symbol] = {'error': 'Collection failed'}
            
            logger.info(f"Refreshed data for {len(symbols)} symbols in {time.time() - start_time:.2f}s")
            return {
                'results': refresh_results,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return {'error': str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        cache_stats = self.cache.get_stats()
        
        return {
            'collection_stats': self.collection_stats,
            'cache_stats': cache_stats,
            'active_collections': len(self.active_collections),
            'executor_stats': {
                'max_workers': self.executor._max_workers,
                'active_threads': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            start_time = time.time()
            
            # Test market data service
            test_quote = self.market_service.get_stock_quote("AAPL")
            market_healthy = test_quote is not None
            
            # Test cache service
            cache_stats = self.cache.get_stats()
            cache_healthy = cache_stats.get('status') == 'connected'
            
            response_time = time.time() - start_time
            overall_healthy = market_healthy and cache_healthy
            
            return {
                'service': 'data_collector',
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'components': {
                    'market_data_service': 'healthy' if market_healthy else 'unhealthy',
                    'cache_service': 'healthy' if cache_healthy else 'unhealthy'
                },
                'response_time': round(response_time, 3),
                'stats': self.get_collection_stats(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'data_collector',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def close(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.market_service:
            self.market_service.close()

# Global collector instance
_data_collector = None

def get_data_collector() -> DataCollector:
    """Get global data collector instance"""
    global _data_collector
    if _data_collector is None:
        _data_collector = DataCollector()
    return _data_collector

# Convenience functions for common operations
async def get_dashboard_data(user_id: str, 
                           portfolio_holdings: List[Dict] = None,
                           watchlist_symbols: List[str] = None) -> Optional[DashboardData]:
    """Convenience function to get dashboard data"""
    collector = get_data_collector()
    return await collector.collect_dashboard_data(user_id, portfolio_holdings, watchlist_symbols)

async def get_stock_data(symbol: str, include_analysis: bool = False) -> Dict[str, Any]:
    """Convenience function to get stock data"""
    collector = get_data_collector()
    if include_analysis:
        return await collector.collect_stock_analysis(symbol)
    else:
        return await collector.collect_stock_data(symbol)

async def refresh_portfolio(user_id: str, holdings: List[Dict]) -> Optional[PortfolioData]:
    """Convenience function to refresh portfolio data"""
    collector = get_data_collector()
    return await collector.collect_portfolio_data(user_id, holdings)

if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_data_collector():
        collector = DataCollector()
        
        print("Testing Data Collector...")
        
        # Test stock data collection
        stock_data = await collector.collect_stock_data("AAPL", include_company_info=True)
        print(f"Stock data collected: {stock_data.get('symbol')} - ${stock_data.get('quote', {}).get('price', 'N/A')}")
        
        # Test portfolio data collection
        sample_holdings = [
            {'symbol': 'AAPL', 'quantity': 10, 'avg_cost': 150.0},
            {'symbol': 'MSFT', 'quantity': 5, 'avg_cost': 300.0}
        ]
        
        portfolio = await collector.collect_portfolio_data("test_user", sample_holdings)
        if portfolio:
            print(f"Portfolio value: ${portfolio.total_value:.2f} ({portfolio.day_change_percent:+.2f}%)")
        
        # Test dashboard data collection
        dashboard = await collector.collect_dashboard_data(
            "test_user", 
            sample_holdings, 
            ["AAPL", "MSFT", "GOOGL"]
        )
        
        if dashboard:
            print(f"Dashboard data collected - Market Status: {dashboard.performance_summary.get('market_status')}")
        
        # Health check
        health = collector.health_check()
        print(f"Health Check: {health['status']}")
        
        # Clean up
        collector.close()
    
    # Run the test
    asyncio.run(test_data_collector())