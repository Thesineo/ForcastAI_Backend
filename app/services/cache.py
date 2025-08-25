import json
import logging
import pickle
import base64
from typing import Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client placeholder
redis_client = None

# ✅ Try to connect only if REDIS_URL is set
if getattr(settings, "REDIS_URL", None):
    try:
        import redis
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established successfully")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available, disabling cache: {str(e)}")
        redis_client = None
else:
    logger.info("ℹ️ No REDIS_URL provided, caching disabled")


def safe_serialize(data: Any) -> str:
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        try:
            pickled_data = pickle.dumps(data)
            encoded_data = base64.b64encode(pickled_data).decode("utf-8")
            return json.dumps({"_pickle": encoded_data})
        except Exception as e:
            logger.error(f"Failed to serialize data: {str(e)}")
            return json.dumps({"error": "Failed to serialize", "type": str(type(data))})


def safe_deserialize(data: str) -> Any:
    try:
        parsed = json.loads(data)
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
    if not redis_client:
        logger.debug("Redis not available, skipping cache set")
        return False
    try:
        redis_client.setex(key, ttl, safe_serialize(value))
        return True
    except Exception as e:
        logger.error(f"Cache set failed for {key}: {str(e)}")
        return False


def cache_get(key: str) -> Optional[Any]:
    if not redis_client:
        logger.debug("Redis not available, cache miss")
        return None
    try:
        cached_value = redis_client.get(key)
        return safe_deserialize(cached_value.decode("utf-8")) if cached_value else None
    except Exception as e:
        logger.error(f"Cache get failed for {key}: {str(e)}")
        return None


def cache_delete(key: str) -> bool:
    if not redis_client:
        return False
    try:
        return bool(redis_client.delete(key))
    except Exception as e:
        logger.error(f"Cache delete failed for {key}: {str(e)}")
        return False


def cache_clear_pattern(pattern: str) -> int:
    if not redis_client:
        return 0
    try:
        keys = redis_client.keys(pattern)
        return redis_client.delete(*keys) if keys else 0
    except Exception as e:
        logger.error(f"Cache clear failed for {pattern}: {str(e)}")
        return 0


def get_cache_stats() -> dict:
    if not redis_client:
        return {"status": "disabled", "message": "Redis not available"}
    try:
        info = redis_client.info()
        return {
            "status": "active",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        return {"status": "error", "message": str(e)}


def cache_result(key_prefix: str, ttl: int = 3600):
    """
    Decorator to cache results.
    Falls back to always recomputing if Redis is disabled.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{hash(str(args) + str(sorted(kwargs.items())))}"
            result = cache_get(cache_key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache_set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
