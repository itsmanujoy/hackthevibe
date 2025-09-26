from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any
from config import Config

class LLMService:
    def __init__(self):
        # Initialize Groq LLM
        if Config.GROQ_API_KEY:
            self.groq_llm = ChatGroq(
                groq_api_key=Config.GROQ_API_KEY,
                model_name=Config.GROQ_MODEL
            )
        else:
            self.groq_llm = None
            print("GROQ_API_KEY not found. Groq LLM will not be available.")

        # Initialize Google GenAI LLM
        if Config.GOOGLE_API_KEY:
            self.genai_llm = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=Config.GOOGLE_API_KEY
            )
        else:
            self.genai_llm = None
            print("GOOGLE_API_KEY not found. Google GenAI LLM will not be available.")

    async def get_response(self, prompt: str, system_message: str = None, use_groq: bool = True) -> str:
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))

        try:
            if use_groq and self.groq_llm:
                response = await self.groq_llm.ainvoke(messages)
                return response.content
            elif self.genai_llm: # Fallback or explicit choice for GenAI
                response = await self.genai_llm.ainvoke(messages)
                return response.content
            else:
                return "No LLM configured or available."
        except Exception as e:
            print(f"Error with LLM: {e}")
            return f"An error occurred while processing your request: {e}"

    async def generate_sql_query(self, natural_language_query: str, df_schema: Dict[str, str]) -> str:
        """
        Uses LLM to generate a Pandas/Polars filtering/grouping code snippet
        from a natural language query and DataFrame schema.
        """
        schema_str = "\n".join([f"- {col}: {dtype}" for col, dtype in df_schema.items()])

        system_message = f"""
        You are an AI assistant that translates natural language questions into executable Python pandas code.
        The user will provide a question and the schema of a pandas DataFrame.
        Your task is to generate ONLY the pandas code snippet required to answer the question,
        without any additional explanations or text.
        Assume the DataFrame is named `df`.
        The code should produce a result that directly answers the question or prepares data for visualization.
        For example, if asked for sales trends, return code that groups by date and sums sales.
        If asked for top customers, return code that sorts and takes the head.
        If asked for filtering, return code that filters rows.
        Be precise with column names.
        
        DataFrame Schema:
        {schema_str}
        
        Examples:
        Question: "Show me total sales by region."
        Code: df.groupby('Region')['Sales'].sum().reset_index()
        
        Question: "What is the average age of customers?"
        Code: df['Age'].mean()
        
        Question: "Filter sales for the last quarter of 2023."
        Code: df[df['Date'].dt.quarter == 4 & (df['Date'].dt.year == 2023)]
        
        Question: "Which column has the most missing values?"
        Code: df.isnull().sum().idxmax()
        """
        
        prompt = f"Question: \"{natural_language_query}\"\nCode:"
        
        code = await self.get_response(prompt=prompt, system_message=system_message, use_groq=True)
        return code.strip()