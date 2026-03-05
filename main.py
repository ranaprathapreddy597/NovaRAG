import os
import io
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import List, Dict
import PyPDF2
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "Online", "system": "NovaRAG Enterprise Server by Rana Prathap Reddy"}

active_sessions: Dict[str, dict] = {}

def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_cloud_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ.get("HF_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    if not session_id:
        raise HTTPException(status_code=400, detail="Secure Session ID required.")
        
    try:
        filename = file.filename.lower()
        file_content = await file.read()
        
        # ENTERPRISE FILE LIMIT: 15MB
        if len(file_content) > 15 * 1024 * 1024:
            del file_content
            return {"status": "error", "message": "File exceeds 15MB limit."}

        if session_id not in active_sessions:
            active_sessions[session_id] = {"vector_db": None, "cif_data": ""}

        if filename.endswith(".pdf"):
            active_sessions[session_id]["cif_data"] = "" 
            try:
                pdf_stream = io.BytesIO(file_content)
                reader = PyPDF2.PdfReader(pdf_stream)
                raw_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
                del file_content, pdf_stream, reader
                gc.collect() 
            except Exception:
                return {"status": "error", "message": "Unreadable PDF formatting."}
            
            raw_text = raw_text[:25000] 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(raw_text)
            del raw_text
            gc.collect()

            try:
                cloud_embedder = get_cloud_embeddings()
                active_sessions[session_id]["vector_db"] = FAISS.from_texts(chunks, cloud_embedder)
                return {"status": "success", "message": "Large document vectorized securely."}
            except Exception:
                return {"status": "error", "message": "AI Engine warming up. Please wait 10s and retry."}

        elif filename.endswith(".cif") or filename.endswith(".txt"):
            active_sessions[session_id]["vector_db"] = None 
            active_sessions[session_id]["cif_data"] = file_content.decode("utf-8", errors="ignore")[:20000] 
            del file_content
            gc.collect()
            return {"status": "success", "message": "Structural data isolated securely."}
        else:
            return {"status": "error", "message": "Unsupported file type."}
            
    except Exception as e:
        return {"status": "error", "message": f"Processing Error: {str(e)}"}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[Dict[str, str]]
    use_web: bool
    temperature: float

def search_web(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        return "\n".join([f"- {res['title']}: {res['body']}" for res in results]) if results else "No current data found."
    except:
        return "Search network error."

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        user_data = active_sessions.get(req.session_id, {"vector_db": None, "cif_data": ""})
        v_db = user_data["vector_db"]
        cif_mem = user_data["cif_data"]

        system_prompt = """You are NovaRAG, an elite Enterprise AI engineered entirely by Rana Prathap Reddy.
        CRITICAL GUARDRAILS:
        1. SAFETY: Never generate harmful or malicious code. Protect user privacy.
        2. ISOLATION: You are assisting ONE specific user in a secure session. Answer ONLY based on provided context.
        3. FORMATTING: Use Markdown Tables for data. Use concise bullet points. No massive paragraphs.
        4. DOMAIN: Material Science (CIF analysis), Robotic Process Automation (RPA), and Software Engineering."""
            
        if v_db is not None:
            relevant_docs = v_db.similarity_search(req.message, k=4)
            rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            system_prompt += f"\n\n--- USER DOCUMENT CONTEXT ---\n{rag_context}\nAnswer strictly using this context."

        if cif_mem.strip():
            system_prompt += f"\n\n--- USER CIF DATA ---\n{cif_mem}\nAnalyze these structural parameters."

        if req.use_web:
            system_prompt += f"\n\n--- WEB DATA ---\n{search_web(req.message)}\nSynthesize this."

        messages = [{"role": "system", "content": system_prompt}] + req.history[-6:] + [{"role": "user", "content": req.message}]
        
        client = get_groq_client()
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=req.temperature,
            max_tokens=1500,
            stream=True
        )

        async def response_generator():
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    safe_token = token.replace("\n", "<br>")
                    yield f"data: {safe_token}\n\n"
                    await asyncio.sleep(0.001) 
            yield "data: [DONE]\n\n"
                
        return StreamingResponse(response_generator(), media_type="text/event-stream")
        
    except Exception as e:
        safe_error = f"**[System Error]** {str(e)}".replace("\n", "<br>")
        async def error_generator(): yield f"data: {safe_error}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")