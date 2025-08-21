import json
import redis
from app.core.config import settings
from typing import Any, Optional
import logging
import pickle
import base64

logger = logging.getLogger(__name__)

# Initialize Redis connection
try:
    if settings.REDIS_URL:
        redis_client = redis.from_url(settings.REDIS_URL)
        # Test connection
        redis_client.ping()
        logger.info("Redis connection established successfully")
    else:
        redis_client = None
        logger.warning("No Redis URL provided, caching disabled")
except Exception as e:
    logger.error(f"Redis connection failed: {str(e)}")
    redis_client = None


def safe_serialize(data: Any) -> str:
    """
    Safely serialize data that might contain tuples, numpy arrays, etc.
    """
    try:
        # First try JSON serialization (fastest)
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        try:
            # Fallback to pickle + base64 for complex objects
            pickled_data = pickle.dumps(data)
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            return json.dumps({"_pickle": encoded_data})
        except Exception as e:
            logger.error(f"Failed to serialize data: {str(e)}")
            # Return a simple error representation
            return json.dumps({"error": "Failed to serialize data", "type": str(type(data))})


def safe_deserialize(data: str) -> Any:
    """
    Safely deserialize data that might have been pickled
    """
    try:
        parsed = json.loads(data)
        
        # Check if this was pickled data
        if isinstance(parsed, dict) and "_pickle" in parsed:
            try:
                decoded_data = base64.b64decode(parsed["_pickle"])
                return pickle.loads(decoded_data)
            except Exception as e:
                logger.error(f"Failed to unpickle data: {str(e)}")
                return None
                
        return parsed
    except Exception as e:
        logger.error(f"Failed to deserialize data: {str(e)}")
        return None


def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Set a value in cache with TTL (time to live) in seconds
    """
    if not redis_client:
        logger.debug("Redis not available, skipping cache set")
        return False
        
    try:
        # Serialize the value safely
        serialized_value = safe_serialize(value)
        
        # Set in Redis with TTL
        redis_client.setex(key, ttl, serialized_value)
        logger.debug(f"Cached key: {key} with TTL: {ttl}")
        return True
        
    except Exception as e:
        logger.error(f"Cache set failed for key {key}: {str(e)}")
        return False


def cache_get(key: str) -> Optional[Any]:
    """
    Get a value from cache
    """
    if not redis_client:
        logger.debug("Redis not available, cache miss")
        return None
        
    try:
        # Get from Redis
        cached_value = redis_client.get(key)
        
        if cached_value is None:
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        # Deserialize the value
        deserialized_value = safe_deserialize(cached_value.decode('utf-8'))
        logger.debug(f"Cache hit for key: {key}")
        return deserialized_value
        
    except Exception as e:
        logger.error(f"Cache get failed for key {key}: {str(e)}")
        return None


def cache_delete(key: str) -> bool:
    """
    Delete a key from cache
    """
    if not redis_client:
        logger.debug("Redis not available, skipping cache delete")
        return False
        
    try:
        result = redis_client.delete(key)
        logger.debug(f"Cache delete for key: {key}, result: {result}")
        return bool(result)
        
    except Exception as e:
        logger.error(f"Cache delete failed for key {key}: {str(e)}")
        return False


def cache_clear_pattern(pattern: str) -> int:
    """
    Clear all keys matching a pattern
    """
    if not redis_client:
        logger.debug("Redis not available, skipping pattern clear")
        return 0
        
    try:
        keys = redis_client.keys(pattern)
        if keys:
            deleted = redis_client.delete(*keys)
            logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
            return deleted
        return 0
        
    except Exception as e:
        logger.error(f"Cache pattern clear failed for pattern {pattern}: {str(e)}")
        return 0


def get_cache_stats() -> dict:
    """
    Get Redis cache statistics
    """
    if not redis_client:
        return {"status": "disabled", "message": "Redis not available"}
        
    try:
        info = redis_client.info()
        return {
            "status": "active",
            "connected_clients": info.get('connected_clients', 0),
            "used_memory": info.get('used_memory_human', '0B'),
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0),
            "total_commands_processed": info.get('total_commands_processed', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        return {"status": "error", "message": str(e)}


# Decorators for easy caching
def cache_result(key_prefix: str, ttl: int = 3600):
    """
    Decorator to cache function results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{key_prefix}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache first
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_set(cache_key, result, ttl)
            logger.debug(f"Cached result for function {func.__name__}")
            
            return result
        return wrapper
    return decorator


# Example usage functions for testing
def test_cache():
    """
    Test cache functionality
    """
    print("🧪 Testing cache functionality...")
    
    # Test basic operations
    test_data = {
        "string": "Hello World",
        "number": 42,
        "list": [1, 2, 3, 4, 5],
        "dict": {"nested": {"value": "test"}},
        "boolean": True,
        "null": None
    }
    
    # Test complex data that might cause JSON issues
    complex_data = {
        "tuple": (1, 2, 3),  # This would cause the original error
        "nested_tuple": {"data": (4, 5, 6)},
        "mixed": [1, "string", (7, 8, 9), {"key": "value"}]
    }
    
    print("Testing basic data...")
    cache_set("test_basic", test_data, 60)
    retrieved_basic = cache_get("test_basic")
    print(f"✅ Basic data test: {'PASS' if retrieved_basic == test_data else 'FAIL'}")
    
    print("Testing complex data with tuples...")
    cache_set("test_complex", complex_data, 60)
    retrieved_complex = cache_get("test_complex")
    print(f"✅ Complex data test: {'PASS' if retrieved_complex is not None else 'FAIL'}")
    
    # Test cache stats
    stats = get_cache_stats()
    print(f"📊 Cache stats: {stats}")
    
    # Clean up
    cache_delete("test_basic")
    cache_delete("test_complex")
    
    print("✅ Cache test completed!")


if __name__ == "__main__":
    test_cache()