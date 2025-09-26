from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """Incoming user query."""
    session_id: Optional[str] = None
    query: str
    dataframe_id: Optional[str] = None
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response returned by the chatbot."""
    session_id: Optional[str] = None
    answer: str
    chart: Optional[str] = None  # base64 chart image URI (if generated)
    metadata: Optional[dict] = None
    quality_report: Optional[str] = None
    intermediate_steps: Optional[List[str]] = None
