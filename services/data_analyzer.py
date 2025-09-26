import pandas as pd
from typing import Dict, Any, List, Tuple
import io
import contextlib
import traceback
from typing import Any, Dict, List, Optional, Tuple


class DataAnalyzer:
    def __init__(self):
        pass

    def get_metadata_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes schema, column types, and provides a metadata summary."""
        metadata = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "column_info": []
        }
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "unique_values": df[col].nunique(),
                "missing_values_count": df[col].isnull().sum(),
                "missing_values_percent": round((df[col].isnull().sum() / df.shape[0]) * 100, 2) if df.shape[0] > 0 else 0
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std()
                })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                 col_info.update({
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max())
                })
            elif pd.api.types.is_string_dtype(df[col]):
                col_info.update({
                    "most_frequent": df[col].mode()[0] if not df[col].mode().empty else None
                })

            metadata["column_info"].append(col_info)
        return metadata

    def run_data_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs built-in data quality checks."""
        quality_report = {}

        # Null Values
        null_counts = df.isnull().sum()
        quality_report['null_values'] = {col: int(count) for col, count in null_counts.items() if count > 0}
        quality_report['total_null_values'] = int(null_counts.sum())

        # Duplicates
        num_duplicates = df.duplicated().sum()
        quality_report['duplicate_rows'] = int(num_duplicates)

        # Outliers (simple IQR method for numeric columns)
        outliers_info = {}
        for col in df.select_dtypes(include=['number']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not col_outliers.empty:
                outliers_info[col] = {
                    "count": int(len(col_outliers)),
                    "examples": col_outliers.head(5).tolist()
                }
        quality_report['outliers_iqr'] = outliers_info

        # Format mismatches (basic check for non-numeric in numeric cols)
        format_issues = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
                # If a column is numeric but has nulls, and after converting to numeric
                # there are still non-numeric representations, that's a format issue.
                # This check is a bit tricky, often handled during initial type conversion.
                # A more robust check might involve inferring types and then checking deviations.
                # For now, a simpler approach: check if original non-numeric values exist
                # that couldn't be coerced.
                pass # More complex type inference needed here
        quality_report['format_mismatches'] = format_issues # Placeholder

        return quality_report

    def execute_code_on_dataframe(self, df: pd.DataFrame, code_snippet: str) -> Tuple[Any, Optional[str]]:
        """
        Executes a pandas code snippet on a DataFrame.
        Captures print statements and returns the result of the last expression.
        """
        local_vars = {'df': df, 'pd': pd}
        output_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_capture):
                # Use exec for multi-line statements, eval for single expressions
                # This tries to run as exec, and if it's a single expression, result will be set
                exec(code_snippet, globals(), local_vars)
                result = local_vars.get('result', None) # If code sets `result = ...`
                if result is None:
                    # Attempt to evaluate if it's a single expression
                    result = eval(code_snippet, globals(), local_vars)
            
            stdout_output = output_capture.getvalue().strip()
            return result, stdout_output
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error executing code: {e}\n{error_traceback}")
            return None, f"Error executing code: {e}\n{error_traceback}"