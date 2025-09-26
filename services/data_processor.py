import os
import pandas as pd
import polars as pl # Optional, for performance with large datasets
from typing import Union, Dict, Any, Optional
from fastapi import UploadFile
from unstructured.partition.auto import partition
from config import Config

class DataProcessor:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER

    async def _save_upload_file(self, upload_file: UploadFile) -> str:
        file_location = os.path.join(self.upload_folder, upload_file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await upload_file.read())
        return file_location

    async def process_file(self, upload_file: UploadFile) -> Union[pd.DataFrame, str, Dict[str, Any], None]:
        file_path = await self._save_upload_file(upload_file)
        file_extension = os.path.splitext(upload_file.filename)[1].lower()

        try:
            if file_extension in ['.csv', '.txt']:
                # Try pandas first, then polars if installed
                try:
                    df = pd.read_csv(file_path)
                    return df
                except ImportError:
                    df = pl.read_csv(file_path)
                    return df.to_pandas() # Convert to pandas for consistency if using polars
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    df = pd.read_excel(file_path)
                    return df
                except ImportError:
                    df = pl.read_excel(file_path)
                    return df.to_pandas()
            elif file_extension == '.json':
                try:
                    df = pd.read_json(file_path)
                    return df
                except ImportError:
                    df = pl.read_json(file_path)
                    return df.to_pandas()
            elif file_extension == '.parquet':
                try:
                    df = pd.read_parquet(file_path)
                    return df
                except ImportError:
                    df = pl.read_parquet(file_path)
                    return df.to_pandas()
            elif file_extension in ['.pdf', '.docx', '.doc']: # Handle documents
                elements = partition(filename=file_path)
                # unstructured returns a list of elements. Join them into a single string.
                document_content = "\n\n".join([str(el) for el in elements])
                return document_content
            else:
                # Fallback for other file types or unsupported ones
                return None
        except Exception as e:
            print(f"Error processing file {upload_file.filename}: {e}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Provides a basic summary of the dataframe."""
        summary = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict(),
            "first_5_rows": df.head().to_dict(orient='records')
        }
        return summary