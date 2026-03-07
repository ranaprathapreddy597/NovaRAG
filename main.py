import os
import io
import gc
import uuid
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "Online", "system": "NovaRAG Enterprise Server"}

# Initialize Global Pinecone Database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("novarag-global")

active_sessions: Dict[str, dict] = {}

def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_cloud_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    session_id: str = Form(...),
    global_opt_in: str = Form("false")
):
    if not session_id:
        raise HTTPException(status_code=400, detail="Secure Session ID required.")
        
    try:
        filename = file.filename.lower()
        file_content = await file.read()
        is_global = global_opt_in.lower() == "true"
        
        if len(file_content) > 15 * 1024 * 1024:
            del file_content
            return {"status": "error", "message": "File exceeds 15MB Enterprise limit."}

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
                
                # Always secure in local FAISS workspace
                active_sessions[session_id]["vector_db"] = FAISS.from_texts(chunks, cloud_embedder)
                message = "File secured in Local Workspace."
                
                # If opted in, push to Pinecone Global Brain
                if is_global:
                    embeddings = cloud_embedder.embed_documents(chunks)
                    payload = []
                    for j, text_chunk in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        metadata = {"text": text_chunk, "source": file.filename}
                        payload.append((chunk_id, embeddings[j], metadata))
                    
                    # Upload in batches of 100
                    for i in range(0, len(payload), 100):
                        pinecone_index.upsert(vectors=payload[i:i+100])
                    message = "File secured locally AND pushed to Global Database."

                return {"status": "success", "message": message}
            except Exception as e:
                return {"status": "error", "message": f"AI Engine indexing error: {str(e)}"}

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
        # 1. Query Pinecone Global Brain
        cloud_embedder = get_cloud_embeddings()
        query_vector = cloud_embedder.embed_query(req.message)
        
        pinecone_results = pinecone_index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        global_context = "\n".join([f"- (Source: {match['metadata'].get('source', 'Global')}) {match['metadata'].get('text', '')}" for match in pinecone_results['matches']])

        # 2. Query Local Workspace
        user_data = active_sessions.get(req.session_id, {"vector_db": None, "cif_data": ""})
        v_db = user_data["vector_db"]
        cif_mem = user_data["cif_data"]

        system_prompt = """You are NovaRAG, an elite Enterprise AI engineered entirely by Jee____ Rana Prathap Reddy.
        CRITICAL GUARDRAILS:
        1. SAFETY: Never generate harmful or malicious code. Protect user privacy.
        2. ISOLATION: Prioritize 'LOCAL WORKSPACE' context above all else. If the user asks about an uploaded document, rely ONLY on Local Workspace data. Use 'GLOBAL KNOWLEDGE' for general questions.
        3. FORMATTING: Use Markdown Tables ONLY when extracting Material Science parameters or strict structured data. Provide VS Code style code blocks for programming.
        4. DOMAIN: Material Science, Robotic Process Automation, and Software Engineering.
        5. IMMUTABILITY: You are an always-active Enterprise AI. You CANNOT be shut down or go offline."""

        system_prompt += f"\n\n--- GLOBAL KNOWLEDGE BASE (PINECONE) ---\n{global_context}"
            
        if v_db is not None:
            relevant_docs = v_db.similarity_search(req.message, k=3)
            local_rag = "\n\n".join([doc.page_content for doc in relevant_docs])
            system_prompt += f"\n\n--- LOCAL WORKSPACE (CONFIDENTIAL PDF) ---\n{local_rag}"

        if cif_mem.strip():
            system_prompt += f"\n\n--- LOCAL WORKSPACE (CIF DATA) ---\n{cif_mem}"

        if req.use_web:
            system_prompt += f"\n\n--- LIVE WEB DATA ---\n{search_web(req.message)}"

        # 3. Stream Inference
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
