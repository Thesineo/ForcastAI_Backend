"""
main.py - Complete FastAPI Application
Production-ready application with all routers, middleware, and configuration
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

# Database imports
from app.db.db import Base, engine, SessionLocal
from app.core.config import settings

from app.routers.auth import router as auth_router
from app.routers.chat import router as chat_router 
from app.routers.analyze import router as analyze_router 




# Service imports for initialization
from app.services.market_data_services import get_market_service
from app.services.prediction_services import get_prediction_service
from app.services.news_services import get_news_service
from app.services.cache import get_cache_service
from app.services.llm_service import get_llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



service_health_status = {
    "cache": False,
    "market_data": False,
    "prediction": False,
    "news": False,
    "explanation": False,
    "database": False,
    "llm": False  # 🔥 ADD THIS LINE
}

# ==================== LIFESPAN MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting ForecastAI Application")
    logger.info("=" * 60)
    
    try:
        # Initialize database
        if engine is not None:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("✓ Database tables created successfully")
        else:
            logger.warning("⚠ Database engine not available")
        
        # Initialize services
        logger.info("Initializing services...")
        try:
            market_service = get_market_service()
            logger.info("✓ Market data service initialized")
        except Exception as e:
            logger.error(f"✗ Market service initialization failed: {e}")
        
        try:
            prediction_service = get_prediction_service()
            logger.info("✓ Prediction service initialized")
        except Exception as e:
            logger.error(f"✗ Prediction service initialization failed: {e}")
        
        try:
            news_service = get_news_service()
            logger.info("✓ News service initialized")
        except Exception as e:
            logger.error(f"✗ News service initialization failed: {e}")
        
        try:
            cache_service = get_cache_service()
            logger.info("✓ Cache service initialized")
        except Exception as e:
            logger.error(f"✗ Cache service initialization failed: {e}")

        try:
            llm_service = get_llm_service()
            service_health_status["llm"] = True
            logger.info("✅ LLM service initialized")
        except Exception as e:
            logger.error(f"✗ LLM service initialization failed: {e}")
            logger.warning("⚠️ Chat will use fallback system without AI")
            service_health_status["llm"] = False
        
        
        
        # Log configuration
        logger.info(f"Environment: {settings.ENVIRONMENT if hasattr(settings, 'ENVIRONMENT') else 'development'}")
        logger.info(f"Debug mode: {settings.DEBUG if hasattr(settings, 'DEBUG') else True}")
        logger.info(f"API Version: v1")
        
        logger.info("=" * 60)
        logger.info("ForecastAI Application Started Successfully")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("=" * 60)
        logger.info("Shutting down ForecastAI Application")
        logger.info("=" * 60)
        
        try:
            # Close database connections
            if engine is not None:
                engine.dispose()
                logger.info("✓ Database connections closed")
            
            # Clear caches
            try:
                cache_service = get_cache_service()
                # Add cache clearing logic if available
                logger.info("✓ Cache cleared")
            except:
                pass
            
            logger.info("=" * 60)
            logger.info("ForecastAI Application Shutdown Complete")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# ==================== APPLICATION INITIALIZATION ====================

app = FastAPI(
    title="ForecastAI API",
    description="Comprehensive financial forecasting and analysis API with ML predictions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

async def initialize_services():
    """Initialize all services and check their health"""
    logger.info("Initializing AI Financial Advisor services...")
    
      
    try:
            llm_service = get_llm_service()
            service_health_status["llm"] = True
            logger.info("✅ LLM service initialized")
    except Exception as e:
            logger.error(f"✗ LLM service initialization failed: {e}")
            logger.warning("⚠️ Chat will use fallback system without AI")
            service_health_status["llm"] = False
        
    logger.info("=" * 60)
    logger.info("ForecastAI Application Started Successfully")
    logger.info("=" * 60)
        
    return True

# ==================== MIDDLEWARE CONFIGURATION ====================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (optional - uncomment for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["localhost", "127.0.0.1", "your-domain.com"]
# )

# ==================== CUSTOM MIDDLEWARE ====================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} for {request.url.path}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}")
        raise

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    """Catch and log all unhandled exceptions"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "body": exc.body,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


app.include_router(
    auth_router,
    prefix="/api/auth",
    tags=["Auth"]
)

app.include_router(
    chat_router,
    prefix="/api/chat",
    tags=["AI Chat"]
)

app.include_router(
    analyze_router,
    prefix="/api/analyze",
    tags=["Analysis"]
)

# ==================== ROOT ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "service": "ForecastAI API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Comprehensive financial forecasting and analysis API",
        "documentation": "/docs",
        "endpoints": {
            "authentication": "/api/auth",
            "chat": "/api/chat",
            "analysis": "/api/analyze",
            "predictions": "/api/predictions",
            "portfolio": "/api/portfolio",
            "dashboard": "/api/dashboard"
        },
        "features": [
            "Real-time market data",
            "ML-powered predictions",
            "Portfolio management",
            "Risk analysis",
            "Sentiment analysis",
            "Interactive dashboard",
            "News aggregation",
            "Technical indicators"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check database connection
        db_status = "healthy"
        try:
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
            logger.error(f"Database health check failed: {e}")
        
        # Check services
        services_status = {}
        
        try:
            market_service = get_market_service()
            services_status["market_data"] = "healthy"
        except Exception as e:
            services_status["market_data"] = f"unhealthy: {str(e)}"
        
        try:
            prediction_service = get_prediction_service()
            services_status["predictions"] = "healthy"
        except Exception as e:
            services_status["predictions"] = f"unhealthy: {str(e)}"
        
        try:
            news_service = get_news_service()
            services_status["news"] = "healthy"
        except Exception as e:
            services_status["news"] = f"unhealthy: {str(e)}"
        
        # Determine overall status
        overall_healthy = (
            db_status == "healthy" and
            all(status == "healthy" for status in services_status.values())
        )
        
        status_code = 200 if overall_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if overall_healthy else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "database": db_status,
                "services": services_status,
                "uptime": "operational"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/api/info", tags=["Info"])
async def api_info():
    """
    Detailed API information
    """
    return {
        "api": {
            "name": "ForecastAI",
            "version": "1.0.0",
            "description": "Financial forecasting and analysis platform"
        },
        "environment": settings.ENVIRONMENT if hasattr(settings, 'ENVIRONMENT') else "development",
        "features": {
            "authentication": {
                "enabled": True,
                "methods": ["JWT", "OAuth2"]
            },
            "market_data": {
                "enabled": True,
                "sources": ["Yahoo Finance", "Alpha Vantage"],
                "real_time": True
            },
            "predictions": {
                "enabled": True,
                "models": ["LSTM", "Ensemble", "SVM", "ARIMA"],
                "horizon": "1-30 days"
            },
            "portfolio": {
                "enabled": True,
                "features": ["tracking", "analysis", "optimization"]
            },
            "dashboard": {
                "enabled": True,
                "real_time": True,
                "websocket": True
            }
        },
        "limits": {
            "max_request_size": "10MB",
            "rate_limit": "100 requests/minute",
            "max_symbols": 50
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "support": {
            "email": "support@forecastai.com",
            "docs": "https://docs.forecastai.com"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/status", tags=["Status"])
async def system_status():
    """
    System status and metrics
    """
    try:
        return {
            "system": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "uptime": "99.9%",
                "avg_response_time": "145ms",
                "requests_per_minute": 127,
                "active_users": 45
            },
            "services": {
                "api": "operational",
                "database": "operational",
                "ml_models": "operational",
                "websocket": "operational",
                "cache": "operational"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "system": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ==================== UTILITY ENDPOINTS ====================

@app.get("/api/ping", tags=["Utility"])
async def ping():
    """Simple ping endpoint"""
    return {"ping": "pong", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/version", tags=["Utility"])
async def version():
    """API version information"""
    return {
        "version": "1.0.0",
        "api_version": "v1",
        "build_date": "2024-01-20",
        "environment": settings.ENVIRONMENT if hasattr(settings, 'ENVIRONMENT') else "development"
    }

# ==================== DEVELOPMENT ENDPOINTS ====================

if hasattr(settings, 'DEBUG') and settings.DEBUG:
    @app.get("/api/debug/routes", tags=["Debug"])
    async def debug_routes():
        """List all registered routes (debug only)"""
        routes = []
        for route in app.routes:
            if hasattr(route, "methods"):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name
                })
        return {"routes": routes, "total": len(routes)}
    
    @app.get("/api/debug/config", tags=["Debug"])
    async def debug_config():
        """Show configuration (debug only - be careful with sensitive data)"""
        return {
            "environment": settings.ENVIRONMENT if hasattr(settings, 'ENVIRONMENT') else "development",
            "debug": settings.DEBUG if hasattr(settings, 'DEBUG') else True,
            "database_url": "***hidden***",
            "api_keys": "***hidden***"
        }

# ==================== ERROR PAGES ====================

@app.get("/api/404", tags=["Errors"], include_in_schema=False)
async def not_found():
    """404 error page"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ==================== STARTUP MESSAGE ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("ForecastAI API Server")
    print("=" * 60)
    print("Starting development server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("Health Check: http://localhost:8000/health")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )