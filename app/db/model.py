from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, func
from app.db.db import Base


class AnalysisLog(Base):
   __tablename__ = "analysis_logs"
   id = Column(Integer, primary_key=True, index=True)
   ticker = Column(String, index=True)
   model_used = Column(String)
   predicted = Column(Float)
   action = Column(String)
   indicators = Column(JSON)
   sentiment = Column(JSON)
   created_at = Column(DateTime(timezone=True), server_default=func.now())


