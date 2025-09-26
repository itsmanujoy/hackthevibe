import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

class VizGenerator:
    def __init__(self):
        pass

    def _plot_to_base64(self, fig_or_ax) -> str:
        """Converts a matplotlib figure or axis to a base64 encoded PNG string."""
        buf = io.BytesIO()
        if isinstance(fig_or_ax, plt.Figure):
            fig_or_ax.savefig(buf, format='png', bbox_inches='tight')
        elif isinstance(fig_or_ax, plt.Axes):
            fig_or_ax.figure.savefig(buf, format='png', bbox_inches='tight')
        else:
            return "" # Handle plotly or other types later
        
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def generate_chart(self, df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None, **kwargs) -> Optional[str]:
        """
        Generates a chart based on specified type and columns, returns base64 image.
        """
        plt.figure(figsize=(10, 6))
        try:
            if chart_type == "bar":
                if x_col and y_col:
                    sns.barplot(x=df[x_col], y=df[y_col], **kwargs)
                elif x_col: # Count plot
                    sns.countplot(x=df[x_col], **kwargs)
                else:
                    return None # Need at least x_col for a bar chart
                plt.title(f"Bar Chart of {y_col if y_col else 'Count'} by {x_col}")

            elif chart_type == "line":
                if x_col and y_col:
                    sns.lineplot(x=df[x_col], y=df[y_col], **kwargs)
                    plt.title(f"Line Chart of {y_col} over {x_col}")

            elif chart_type == "scatter":
                if x_col and y_col:
                    sns.scatterplot(x=df[x_col], y=df[y_col], **kwargs)
                    plt.title(f"Scatter Plot of {y_col} vs {x_col}")

            elif chart_type == "hist":
                if x_col:
                    sns.histplot(df[x_col], kde=True, **kwargs)
                    plt.title(f"Histogram of {x_col}")

            elif chart_type == "box":
                if x_col:
                    sns.boxplot(y=df[x_col], **kwargs)
                    plt.title(f"Box Plot of {x_col}")
                if x_col and y_col: # Box plot by category
                    sns.boxplot(x=df[x_col], y=df[y_col], **kwargs)
                    plt.title(f"Box Plot of {y_col} by {x_col}")
            
            else:
                return None # Unsupported chart type

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            base64_img = self._plot_to_base64(plt.gcf())
            plt.close() # Close the plot to free memory
            return base64_img

        except Exception as e:
            print(f"Error generating {chart_type} chart: {e}")
            plt.close()
            return None

    def generate_plotly_chart(self, df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generates an interactive Plotly chart as a JSON object.
        """
        fig = None
        try:
            if chart_type == "bar":
                if x_col and y_col:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}", **kwargs)
                elif x_col:
                    # Count bar chart
                    counts = df[x_col].value_counts().reset_index()
                    counts.columns = [x_col, 'count']
                    fig = px.bar(counts, x=x_col, y='count', title=f"Count of {x_col}", **kwargs)

            elif chart_type == "line":
                if x_col and y_col:
                    fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} over {x_col}", **kwargs)

            elif chart_type == "scatter":
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}", **kwargs)

            elif chart_type == "hist":
                if x_col:
                    fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}", **kwargs)

            elif chart_type == "box":
                if x_col:
                    fig = px.box(df, y=x_col, title=f"Box Plot of {x_col}", **kwargs)
                if x_col and y_col:
                    fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}", **kwargs)
            
            if fig:
                return fig.to_json()
            else:
                return None

        except Exception as e:
            print(f"Error generating Plotly {chart_type} chart: {e}")
            return None