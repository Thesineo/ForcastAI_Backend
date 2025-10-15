"""
Enhanced Cache Service for AI Financial Advisor
Handles caching for market data, predictions, news, and user sessions
"""

import redis
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from functools import wraps
import asyncio
import redis.asyncio as aioredis
from redis.asyncio import Redis 



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheType(Enum):
    """Cache type enumeration for different data categories"""
    MARKET_DATA = "market_data"
    STOCK_PRICES = "stock_prices"
    PREDICTIONS = "predictions"
    NEWS = "news"
    SENTIMENT = "sentiment"
    USER_SESSION = "user_session"
    CHAT_HISTORY = "chat_history"
    PORTFOLIO = "portfolio"
    WATCHLIST = "watchlist"
    INDICATORS = "indicators"
    ANALYSIS = "analysis"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 10
    
    # TTL settings for different cache types (in seconds)
    ttl_settings: Dict[CacheType, int] = None
    
    def __post_init__(self):
        if self.ttl_settings is None:
            self.ttl_settings = {
                CacheType.MARKET_DATA: 300,      # 5 minutes
                CacheType.STOCK_PRICES: 60,       # 1 minute
                CacheType.PREDICTIONS: 1800,      # 30 minutes
                CacheType.NEWS: 900,              # 15 minutes
                CacheType.SENTIMENT: 1800,        # 30 minutes
                CacheType.USER_SESSION: 86400,    # 24 hours
                CacheType.CHAT_HISTORY: 604800,   # 1 week
                CacheType.PORTFOLIO: 300,         # 5 minutes
                CacheType.WATCHLIST: 3600,        # 1 hour
                CacheType.INDICATORS: 900,        # 15 minutes
                CacheType.ANALYSIS: 1800,         # 30 minutes
            }

class CacheService:
    """Enhanced Redis-based cache service"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.async_redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                max_connections=self.config.max_connections
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _get_async_redis(self):
        """Get async Redis client"""
        if not self.async_redis_client:
            try:
                self.async_redis_client = await aioredis.from_url(
                    f"redis://{self.config.redis_host}:{self.config.redis_port}",
                    password=self.config.redis_password,
                    db=self.config.redis_db
                )
            except Exception as e:
                logger.error(f"Failed to create async Redis client: {e}")
                return None
        return self.async_redis_client
    
    def _generate_key(self, cache_type: CacheType, identifier: str, **kwargs) -> str:
        """Generate cache key with proper namespace"""
        namespace = f"ai_advisor:{cache_type.value}"
        
        # Add additional parameters to key if provided
        if kwargs:
            key_data = f"{identifier}:{':'.join(f'{k}_{v}' for k, v in sorted(kwargs.items()))}"
        else:
            key_data = identifier
            
        # Create hash for very long keys
        if len(key_data) > 100:
            key_data = hashlib.md5(key_data.encode()).hexdigest()
            
        return f"{namespace}:{key_data}"
    
    def set(self, cache_type: CacheType, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Set cache value with automatic serialization"""
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            ttl = ttl or self.config.ttl_settings.get(cache_type, self.config.default_ttl)
            
            # Serialize complex objects
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, default=str)
            elif hasattr(value, '__dict__'):  # Custom objects
                serialized_value = pickle.dumps(value)
                cache_key += ":pickled"
            else:
                serialized_value = str(value)
            
            success = self.redis_client.setex(cache_key, ttl, serialized_value)
            
            if success:
                logger.debug(f"Cached {cache_type.value}: {key} (TTL: {ttl}s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_type.value}:{key} - {e}")
            return False
    
    def get(self, cache_type: CacheType, key: str, **kwargs) -> Optional[Any]:
        """Get cache value with automatic deserialization"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            value = self.redis_client.get(cache_key)
            
            if value is None:
                return None
            
            # Handle pickled objects
            if cache_key.endswith(":pickled"):
                return pickle.loads(value.encode('latin-1'))
            
            # Try JSON deserialization first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for {cache_type.value}:{key} - {e}")
            return None
    
    def delete(self, cache_type: CacheType, key: str, **kwargs) -> bool:
        """Delete cache entry"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            result = self.redis_client.delete(cache_key)
            logger.debug(f"Deleted cache key: {cache_key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error for {cache_type.value}:{key} - {e}")
            return False
    
    def exists(self, cache_type: CacheType, key: str, **kwargs) -> bool:
        """Check if cache key exists"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            return bool(self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Cache exists check error: {e}")
            return False
    
    def get_ttl(self, cache_type: CacheType, key: str, **kwargs) -> int:
        """Get remaining TTL for cache key"""
        if not self.redis_client:
            return -1
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            return self.redis_client.ttl(cache_key)
        except Exception as e:
            logger.error(f"Cache TTL check error: {e}")
            return -1
    
    def extend_ttl(self, cache_type: CacheType, key: str, additional_time: int, **kwargs) -> bool:
        """Extend TTL of existing cache entry"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            current_ttl = self.redis_client.ttl(cache_key)
            if current_ttl > 0:
                new_ttl = current_ttl + additional_time
                return bool(self.redis_client.expire(cache_key, new_ttl))
            return False
        except Exception as e:
            logger.error(f"Cache TTL extension error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache pattern clear error: {e}")
            return 0
    
    def clear_cache_type(self, cache_type: CacheType) -> int:
        """Clear all cache entries of specific type"""
        pattern = f"ai_advisor:{cache_type.value}:*"
        return self.clear_pattern(pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disconnected"}
        
        try:
            info = self.redis_client.info()
            stats = {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total = hits + misses
            stats["hit_rate"] = round((hits / total * 100) if total > 0 else 0, 2)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def async_get(self, cache_type: CacheType, key: str, **kwargs) -> Optional[Any]:
        """Async version of get method"""
        redis_client = await self._get_async_redis()
        if not redis_client:
            return None
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            value = await redis_client.get(cache_key)
            
            if value is None:
                return None
            
            # Handle different value types
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    async def async_set(self, cache_type: CacheType, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Async version of set method"""
        redis_client = await self._get_async_redis()
        if not redis_client:
            return False
        
        try:
            cache_key = self._generate_key(cache_type, key, **kwargs)
            ttl = ttl or self.config.ttl_settings.get(cache_type, self.config.default_ttl)
            
            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            await redis_client.setex(cache_key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
            return False
    
    def close(self):
        """Close Redis connections"""
        if self.redis_client:
            self.redis_client.close()
        if self.async_redis_client:
            asyncio.create_task(self.async_redis_client.close())

# Cache decorator for automatic caching
def cached(cache_type: CacheType, key_func=None, ttl=None):
    """Decorator for automatic function result caching"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheService()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache first
            cached_result = cache.get(cache_type, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_type, cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        return wrapper
    return decorator

# Specialized cache methods for common use cases
class SpecializedCache:
    """Specialized cache methods for specific data types"""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    def cache_stock_data(self, symbol: str, timeframe: str, data: Dict) -> bool:
        """Cache stock price data"""
        return self.cache.set(
            CacheType.STOCK_PRICES, 
            symbol, 
            data, 
            symbol=symbol, 
            timeframe=timeframe
        )
    
    def get_stock_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached stock data"""
        return self.cache.get(
            CacheType.STOCK_PRICES, 
            symbol, 
            symbol=symbol, 
            timeframe=timeframe
        )
    
    def cache_user_portfolio(self, user_id: str, portfolio_data: Dict) -> bool:
        """Cache user portfolio data"""
        return self.cache.set(CacheType.PORTFOLIO, user_id, portfolio_data)
    
    def get_user_portfolio(self, user_id: str) -> Optional[Dict]:
        """Get cached user portfolio"""
        return self.cache.get(CacheType.PORTFOLIO, user_id)
    
    def cache_chat_history(self, user_id: str, chat_data: List[Dict]) -> bool:
        """Cache chat history"""
        return self.cache.set(CacheType.CHAT_HISTORY, user_id, chat_data)
    
    def get_chat_history(self, user_id: str) -> Optional[List[Dict]]:
        """Get cached chat history"""
        return self.cache.get(CacheType.CHAT_HISTORY, user_id)
    
    def cache_predictions(self, symbol: str, prediction_type: str, predictions: Dict) -> bool:
        """Cache ML predictions"""
        return self.cache.set(
            CacheType.PREDICTIONS, 
            symbol, 
            predictions,
            prediction_type=prediction_type
        )
    
    def get_predictions(self, symbol: str, prediction_type: str) -> Optional[Dict]:
        """Get cached predictions"""
        return self.cache.get(
            CacheType.PREDICTIONS, 
            symbol, 
            prediction_type=prediction_type
        )

# Global cache instance
_cache_service = None

def get_cache_service(config: CacheConfig = None) -> CacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService(config)
    return _cache_service

def get_specialized_cache(config: CacheConfig = None) -> SpecializedCache:
    """Get specialized cache instance"""
    cache_service = get_cache_service(config)
    return SpecializedCache(cache_service)

# Health check function
def health_check() -> Dict[str, Any]:
    """Perform cache service health check"""
    cache = get_cache_service()
    
    try:
        # Test basic operations
        test_key = f"health_check_{datetime.now().timestamp()}"
        test_value = {"status": "ok", "timestamp": datetime.now().isoformat()}
        
        # Test set
        set_success = cache.set(CacheType.ANALYSIS, test_key, test_value, ttl=60)
        
        # Test get
        get_result = cache.get(CacheType.ANALYSIS, test_key)
        
        # Test delete
        delete_success = cache.delete(CacheType.ANALYSIS, test_key)
        
        health_status = {
            "service": "cache",
            "status": "healthy" if (set_success and get_result and delete_success) else "unhealthy",
            "operations": {
                "set": set_success,
                "get": bool(get_result),
                "delete": delete_success
            },
            "stats": cache.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        return {
            "service": "cache",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Example usage and testing
    cache = get_cache_service()
    specialized = get_specialized_cache()
    
    # Test basic operations
    print("Testing cache operations...")
    
    # Test stock data caching
    sample_stock_data = {
        "price": 150.25,
        "change": 2.15,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat()
    }
    
    specialized.cache_stock_data("AAPL", "1d", sample_stock_data)
    cached_data = specialized.get_stock_data("AAPL", "1d")
    print(f"Stock data cached and retrieved: {cached_data}")
    
    # Health check
    health = health_check()
    print(f"Cache health check: {health}")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")