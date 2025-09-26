from pydantic import BaseModel
from typing import Optional
import pandas as pd


class UploadDataRequest(BaseModel):
    """Request schema for uploading a dataset."""
    file_name: str
    file_type: str  # csv, xlsx, parquet, json, etc.
    dataframe_id: Optional[str] = None


class UploadDataResponse(BaseModel):
    """Response after uploading a dataset."""
    dataframe_id: str
    columns: list[str]
    n_rows: int
    preview: list[dict]  # small preview for UI


class UploadDocRequest(BaseModel):
    """Request schema for uploading a document for RAG/QA."""
    file_name: str
    document_id: Optional[str] = None


class UploadDocResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str
    n_chunks: int
