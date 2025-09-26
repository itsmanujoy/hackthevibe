# ### Part 8: FastAPI Backend (`main.py`)

# Now, let's put it all together in your main FastAPI application.

# **`main.py`**

# ```python
# from fastapi import FastAPI, UploadFile, File, Form, HTTPExceptio
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import Optional, List, Dict, Any
# import uvicorn
# import uuid # For unique document IDs
# import pandas as pd # For type hinting DataFrame in memory

# # Import your services and agent
# from config import Config
# from services.data_processor import DataProcessor
# from services.llm_service import LLMService
# from services.document_qa import DocumentQAService
# from services.data_analyzer import DataAnalyzer
# from services.viz_generator import VizGenerator
# from agents.main_agent import InsightBotAgent

# # --- Pydantic Models for API Requests/Responses ---
# class ChatRequest(BaseModel):
#     user_message: str
#     session_id: str
#     chat_history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}]


# class ChatResponse(BaseModel):
#     response: str
#     image_data: Optional[str] = None # Base64 encoded image
#     chart_json: Optional[Dict[str, Any]] = None # Plotly chart JSON
#     # Add other response types as needed

# # --- FastAPI App Setup ---
# app = FastAPI(
#     title="InsightBot - Interactive Data Analyzer",
#     description="AI-powered data analyzer and conversational assistant.",
#     version="0.1.0"
# )

# # --- Initialize Services and Agent (Global or per request for simplicity here) ---
# # For a production app, you might use dependency injection or a factory.
# llm_service = LLMService()
# doc_qa_service = DocumentQAService(llm_service) # doc_qa needs LLM
# data_analyzer = DataAnalyzer()
# viz_generator = VizGenerator()
# data_processor = DataProcessor()

# # Store active sessions and their associated data/documents/agents
# # In a real app, this would be a persistent store (database, Redis, etc.)
# # For this example, in-memory dictionary.
# active_sessions: Dict[str, Dict[str, Any]] = {}
# """
# active_sessions = {
#     "session_id_xyz": {
#         "dataframe": pd.DataFrame,
#         "document_id": "doc_abc",
#         "agent": InsightBotAgent_instance,
#         "chat_history": []
#     }
# }
# """

# def get_or_create_session(session_id: str) -> Dict[str, Any]:
#     """Retrieves or creates a session for a given session_id."""
#     if session_id not in active_sessions:
#         # Create a new agent for the session
#         session_agent = InsightBotAgent(llm_service, doc_qa_service, data_analyzer, viz_generator)
#         active_sessions[session_id] = {
#             "dataframe": None,
#             "document_id": None,
#             "agent": session_agent,
#             "chat_history": []
#         }
#     return active_sessions[session_id]

# # --- API Endpoints ---

# @app.post("/upload-data/")
# async def upload_data(session_id: str = Form(...), file: UploadFile = File(...)):
#     """
#     Uploads a dataset (CSV, Excel, JSON, Parquet) for analysis.
#     """
#     session = get_or_create_session(session_id)
    
#     print(f"[{session_id}] Received data upload: {file.filename}")
#     processed_content = await data_processor.process_file(file)

#     if isinstance(processed_content, pd.DataFrame):
#         session["dataframe"] = processed_content
#         session["agent"].set_dataframe(processed_content) # Update agent's dataframe
#         df_summary = data_processor.get_data_summary(processed_content)
#         print(f"[{session_id}] Dataframe loaded. Columns: {df_summary['column_names']}")
#         return JSONResponse(content={
#             "message": f"Dataset '{file.filename}' uploaded and ready for analysis.",
#             "data_summary": df_summary
#         })
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file type or error processing data file.")

# @app.post("/upload-document/")
# async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
#     """
#     Uploads a document (PDF, Word) for natural language querying.
#     """
#     session = get_or_create_session(session_id)
    
#     print(f"[{session_id}] Received document upload: {file.filename}")
#     document_content = await data_processor.process_file(file)

#     if isinstance(document_content, str):
#         # Generate a unique ID for this document within the session
#         doc_id = f"{session_id}_{uuid.uuid4().hex}"
#         await doc_qa_service.add_document_for_qa(document_content, doc_id)
#         session["document_id"] = doc_id
#         session["agent"].set_document_context(doc_id) # Update agent's document context
#         print(f"[{session_id}] Document '{file.filename}' processed for QA with ID: {doc_id}")
#         return JSONResponse(content={
#             "message": f"Document '{file.filename}' uploaded and ready for QA.",
#             "document_id": doc_id,
#             "first_500_chars": document_content[:500] + "..." if len(document_content) > 500 else document_content
#         })
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file type or error processing document file.")


# @app.post("/chat/", response_model=ChatResponse)
# async def chat_with_insightbot(request: ChatRequest):
#     """
#     Sends a natural language query to InsightBot.
#     """
#     session = get_or_create_session(request.session_id)
#     agent = session["agent"]
    
#     print(f"[{request.session_id}] User message: {request.user_message}")
    
#     # Update agent's dataframe/document context in case they were updated via separate API calls
#     # This might be redundant if set_dataframe/set_document_context are called on upload
#     # but good for robustness if the agent is long-lived and session data changes.
#     if session["dataframe"] is not None and agent.current_dataframe is not session["dataframe"]:
#         agent.set_dataframe(session["dataframe"])
#     if session["document_id"] is not None and agent.current_document_id is not session["document_id"]:
#         agent.set_document_context(session["document_id"])

#     # Invoke the agent
#     agent_response = await agent.ainvoke_agent(request.user_message, request.chat_history)
    
#     response_text = agent_response["answer"]
#     image_data = None
#     chart_json = None

#     # Check for special tags from the agent output
#     if "Generated image:

#         import { Agent } from "http"; 
#         if response_text.startswith("Generated image: `") and response_text.endswith("`"):
#             image_data = response_text.replace("Generated image: `", "").replace("`", "")
#             response_text = "Here's the visualization you requested:"
#         elif response_text.startswith("Generated interactive Plotly chart (JSON): "):
#             try:
#                 chart_json_str = response_text.replace("Generated interactive Plotly chart (JSON): ", "")
#                 chart_json = json.loads(chart_json_str)
#                 response_text = "Here's the interactive chart you requested:"
#             except json.JSONDecodeError:
#                 print(f"[{request.session_id}] Error parsing Plotly JSON: {chart_json_str}")
#                 response_text = "I generated an interactive chart, but there was an error processing its data. Please try again."

#         # Update chat history for the session
#         session["chat_history"].append({"role": "user", "content": request.user_message})
#         session["chat_history"].append({"role": "assistant", "content": response_text})

#         print(f"[{request.session_id}] Agent response: {response_text[:100]}...") # Log truncated response
#         return ChatResponse(response=response_text, image_data=image_data, chart_json=chart_json)


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import pandas as pd
import json

# Import your services and agent
from config import Config
from services.data_processor import DataProcessor
from services.llm_service import LLMService
from services.document_qa import DocumentQAService
from services.data_analyzer import DataAnalyzer
from services.viz_generator import VizGenerator
from agents.main_agent import InsightBotAgent

# --- Pydantic Models for API Requests/Responses ---
class ChatRequest(BaseModel):
    user_message: str
    session_id: str
    chat_history: List[Dict[str, str]] = []  # [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    response: str
    image_data: Optional[str] = None  # Base64 encoded image
    chart_json: Optional[Dict[str, Any]] = None  # Plotly chart JSON

# --- FastAPI App Setup ---
app = FastAPI(
    title="InsightBot - Interactive Data Analyzer",
    description="AI-powered data analyzer and conversational assistant.",
    version="0.1.0"
)

# --- Initialize Services and Agent ---
llm_service = LLMService()
doc_qa_service = DocumentQAService(llm_service)
data_analyzer = DataAnalyzer()
viz_generator = VizGenerator()
data_processor = DataProcessor()

# In-memory session store
active_sessions: Dict[str, Dict[str, Any]] = {}

def get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in active_sessions:
        session_agent = InsightBotAgent(llm_service, doc_qa_service, data_analyzer, viz_generator)
        active_sessions[session_id] = {
            "dataframe": None,
            "document_id": None,
            "agent": session_agent,
            "chat_history": []
        }
    return active_sessions[session_id]

# --- API Endpoints ---

@app.post("/upload-data/")
async def upload_data(session_id: str = Form(...), file: UploadFile = File(...)):
    session = get_or_create_session(session_id)
    processed_content = await data_processor.process_file(file)

    if isinstance(processed_content, pd.DataFrame):
        session["dataframe"] = processed_content
        session["agent"].set_dataframe(processed_content)
        df_summary = data_processor.get_data_summary(processed_content)
        return JSONResponse(content={
            "message": f"Dataset '{file.filename}' uploaded and ready for analysis.",
            "data_summary": df_summary
        })
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type or error processing data file.")

@app.post("/upload-document/")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    session = get_or_create_session(session_id)
    document_content = await data_processor.process_file(file)

    if isinstance(document_content, str):
        doc_id = f"{session_id}_{uuid.uuid4().hex}"
        await doc_qa_service.add_document_for_qa(document_content, doc_id)
        session["document_id"] = doc_id
        session["agent"].set_document_context(doc_id)
        return JSONResponse(content={
            "message": f"Document '{file.filename}' uploaded and ready for QA.",
            "document_id": doc_id,
            "first_500_chars": document_content[:500] + "..." if len(document_content) > 500 else document_content
        })
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type or error processing document file.")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_insightbot(request: ChatRequest):
    session = get_or_create_session(request.session_id)
    agent = session["agent"]

    # Sync agent context with session
    if session["dataframe"] is not None:
        agent.set_dataframe(session["dataframe"])
    if session["document_id"] is not None:
        agent.set_document_context(session["document_id"])

    agent_response = await agent.ainvoke_agent(request.user_message, request.chat_history)
    
    response_text = agent_response.get("answer", "")
    image_data = None
    chart_json = None

    # Detect generated charts or images from agent output
    if response_text.startswith("Generated image: `") and response_text.endswith("`"):
        image_data = response_text.replace("Generated image: `", "").replace("`", "")
        response_text = "Here's the visualization you requested:"
    elif response_text.startswith("Generated interactive Plotly chart (JSON): "):
        try:
            chart_json_str = response_text.replace("Generated interactive Plotly chart (JSON): ", "")
            chart_json = json.loads(chart_json_str)
            response_text = "Here's the interactive chart you requested:"
        except json.JSONDecodeError:
            response_text = "I generated an interactive chart, but there was an error processing its data. Please try again."

    # Update session chat history
    session["chat_history"].append({"role": "user", "content": request.user_message})
    session["chat_history"].append({"role": "assistant", "content": response_text})

    return ChatResponse(response=response_text, image_data=image_data, chart_json=chart_json)

# --- Run Uvicorn (Optional) ---
if __name__ == "__main__":
    # Local development only
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


