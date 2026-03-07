import os
import io
import gc
import uuid
import asyncio
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import PyPDF2
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Local AI Models
from langchain_community.embeddings import HuggingFaceEmbeddings 
from groq import Groq
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "Operational", "tier": "Enterprise Multi-RAG"}

# 1. Initialize Infrastructure securely
try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index("novarag-global")
except Exception as e:
    print(f"⚠️ Pinecone Warning: {e}")
    pinecone_index = None

active_sessions: Dict[str, dict] = {}

def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Local Embeddings for Zero-Latency / Zero-Cost vectorization
def get_cloud_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
                return {"status": "error", "message": "Corrupted or encrypted PDF formatting."}
            
            # Enterprise limit to prevent RAM overflow
            raw_text = raw_text[:30000] 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = text_splitter.split_text(raw_text)
            del raw_text
            gc.collect()

            try:
                local_embedder = get_cloud_embeddings()
                
                # 100% Isolated Local Storage
                active_sessions[session_id]["vector_db"] = FAISS.from_texts(chunks, local_embedder)
                message = "File secured in highly isolated Local Workspace."
                
                # Global Knowledge Contribution
                if is_global and pinecone_index:
                    embeddings = local_embedder.embed_documents(chunks)
                    payload = [(str(uuid.uuid4()), embeddings[j], {"text": chunk, "source": file.filename}) for j, chunk in enumerate(chunks)]
                    
                    for i in range(0, len(payload), 100):
                        pinecone_index.upsert(vectors=payload[i:i+100])
                    message = "File secured locally AND successfully pushed to Global Database."

                return {"status": "success", "message": message}
            except Exception as e:
                return {"status": "error", "message": f"AI Engine indexing error: {str(e)}"}

        elif filename.endswith(".cif") or filename.endswith(".txt"):
            active_sessions[session_id]["vector_db"] = None 
            active_sessions[session_id]["cif_data"] = file_content.decode("utf-8", errors="ignore")[:25000] 
            del file_content
            gc.collect()
            return {"status": "success", "message": "Structural data isolated securely in RAM."}
        else:
            return {"status": "error", "message": "Unsupported file format."}
            
    except Exception as e:
        return {"status": "error", "message": f"Critical Processing Error: {str(e)}"}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[Dict[str, str]]
    use_web: bool
    temperature: float

# --- ASYNC PARALLEL WORKERS ---
async def fetch_pinecone(query_vector):
    if not pinecone_index: return ""
    try:
        # Run synchronous Pinecone code in a background thread so it doesn't block FastAPI
        res = await asyncio.to_thread(pinecone_index.query, vector=query_vector, top_k=3, include_metadata=True)
        return "\n".join([f"- (Source: {m['metadata'].get('source', 'Global')}) {m['metadata'].get('text', '')}" for m in res.get('matches', [])])
    except:
        return ""

async def fetch_faiss(v_db, user_query):
    if not v_db: return ""
    try:
        docs = await asyncio.to_thread(v_db.similarity_search, user_query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    except:
        return ""

async def fetch_web(query, use_web):
    if not use_web: return ""
    try:
        results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=2)))
        return "\n".join([f"- {res['title']}: {res['body']}" for res in results]) if results else ""
    except:
        return ""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        user_query = req.message.strip()
        local_embedder = get_cloud_embeddings()
        
        # We only vectorize the query ONCE to save compute time
        query_vector = local_embedder.embed_query(user_query)

        user_data = active_sessions.get(req.session_id, {"vector_db": None, "cif_data": ""})
        v_db = user_data["vector_db"]
        cif_mem = user_data["cif_data"]

        # 🚀 ENTERPRISE PARALLEL PROCESSING: Fetch all 3 databases simultaneously!
        global_task = fetch_pinecone(query_vector)
        local_task = fetch_faiss(v_db, user_query)
        web_task = fetch_web(user_query, req.use_web)

        global_context, local_rag, web_context = await asyncio.gather(global_task, local_task, web_task)

        # --- THE MASTER ROUTING PROMPT ---
        system_prompt = """You are NovaRAG, an elite Enterprise AI engineered entirely by Jee____ Rana Prathap Reddy.
        You are a highly capable intelligence with vast internal knowledge.

        CRITICAL EXECUTION PROTOCOL:
        1. CONVERSATION DETECTOR: If the user query is a simple greeting (e.g., "hi", "hello"), social pleasantry, or generic question ("how are you"), IGNORE the database context below. Respond naturally, warmly, and briefly.
        2. DATA HIERARCHY: If the user asks a factual, technical, or document-related question, you MUST synthesize your answer using the provided context in this strict order of authority:
           [Tier 1] LOCAL WORKSPACE (Confidential user uploads. This is Absolute Truth.)
           [Tier 2] GLOBAL KNOWLEDGE BASE (Verified Pinecone facts.)
           [Tier 3] LIVE WEB DATA (If enabled.)
           [Tier 4] Your internal Llama-3 parameters (Use only to fill gaps, do not hallucinate numbers).
        3. INTEGRITY: If the databases do not contain the specific mathematical or scientific parameter requested, state "I do not have that exact parameter in the provided documents" rather than guessing.
        4. FORMATTING: Use Markdown Tables for Materials Science parameters, structural coordinates, and RPA mappings. Use VS Code style blocks for programming."""

        if global_context.strip():
            system_prompt += f"\n\n--- [Tier 2] GLOBAL KNOWLEDGE BASE ---\n{global_context}"
            
        if local_rag.strip():
            system_prompt += f"\n\n--- [Tier 1] LOCAL WORKSPACE (CONFIDENTIAL PDF) ---\n{local_rag}"

        if cif_mem.strip():
            system_prompt += f"\n\n--- [Tier 1] LOCAL WORKSPACE (CIF DATA) ---\n{cif_mem}"
            
        if web_context.strip():
            system_prompt += f"\n\n--- [Tier 3] LIVE WEB DATA ---\n{web_context}"

        # 🧠 SLIDING WINDOW MEMORY: Only keep the last 6 messages to prevent context overflow!
        safe_history = req.history[-6:]
        messages = [{"role": "system", "content": system_prompt}] + safe_history + [{"role": "user", "content": user_query}]
        
        client = get_groq_client()
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=req.temperature,
            max_tokens=2000,
            stream=True
        )

        async def response_generator():
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    safe_token = token.replace("\n", "<br>")
                    yield f"data: {safe_token}\n\n"
                    # Ultra-low latency yielding
                    await asyncio.sleep(0.0001) 
            yield "data: [DONE]\n\n"
                
        return StreamingResponse(response_generator(), media_type="text/event-stream")
        
    except Exception as e:
        safe_error = f"**[System Alert]** Graceful Degradation active. Error encountered: {str(e)}".replace("\n", "<br>")
        async def error_generator(): yield f"data: {safe_error}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")