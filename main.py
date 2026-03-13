import os
import io
import gc
import uuid
import time
import asyncio
import requests
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pdfplumber
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from groq import Groq
from pinecone import Pinecone

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ============================================================================
# 🛡️ THE RESILIENT CLOUD EMBEDDER (Zero-Crash Architecture)
# ============================================================================
class ResilientHFEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=self.headers, json={"inputs": inputs}, timeout=30)
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "estimated_time" in data:
                    time.sleep(min(data["estimated_time"], 15))
                    continue
            except Exception:
                time.sleep(2)
        return [[0.0] * 384 for _ in range(len(inputs))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# ============================================================================
# 🧠 GLOBAL INFRASTRUCTURE
# ============================================================================
try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index("novarag-global")
except Exception:
    pinecone_index = None

hf_token = os.environ.get("HF_TOKEN")
global_embedder = ResilientHFEmbeddings(api_key=hf_token)
active_sessions: Dict[str, dict] = {}

def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ============================================================================
# 📂 UPLOAD ENDPOINT (Now with Advanced Table Extraction)
# ============================================================================
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
            return {"status": "error", "message": "File exceeds 15MB limit."}

        if session_id not in active_sessions:
            active_sessions[session_id] = {"vector_db": None, "cif_data": ""}

        if filename.endswith(".pdf"):
            active_sessions[session_id]["cif_data"] = "" 
            try:
                pdf_stream = io.BytesIO(file_content)
                raw_text = ""
                
                # Enterprise Upgrade: pdfplumber extracts exact table rows/columns
                with pdfplumber.open(pdf_stream) as pdf:
                    for page in pdf.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            raw_text += extracted_text + "\n"
                        
                        tables = page.extract_tables()
                        for table in tables:
                            raw_text += "\n--- EXTRACTED DATA TABLE ---\n"
                            for row in table:
                                cleaned_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                                raw_text += " | ".join(cleaned_row) + "\n"
                            raw_text += "---------------------------\n"
                            
                del file_content, pdf_stream
                gc.collect() 
            except Exception as e:
                return {"status": "error", "message": f"PDF parsing error: {str(e)}"}
            
            raw_text = raw_text[:40000] 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
            chunks = text_splitter.split_text(raw_text)
            del raw_text
            gc.collect() 

            try:
                active_sessions[session_id]["vector_db"] = FAISS.from_texts(chunks, global_embedder)
                message = "File secured in Local Workspace. Tables extracted successfully."
                
                if is_global and pinecone_index:
                    embeddings = global_embedder.embed_documents(chunks)
                    payload = [(str(uuid.uuid4()), embeddings[j], {"text": chunk, "source": file.filename}) for j, chunk in enumerate(chunks)]
                    for i in range(0, len(payload), 100):
                        pinecone_index.upsert(vectors=payload[i:i+100])
                    message = "File secured locally AND pushed to Global Database."

                return {"status": "success", "message": message}
            except Exception as e:
                return {"status": "error", "message": f"AI Engine indexing error: {str(e)}"}

        elif filename.endswith(".cif") or filename.endswith(".txt"):
            active_sessions[session_id]["vector_db"] = None 
            active_sessions[session_id]["cif_data"] = file_content.decode("utf-8", errors="ignore")[:25000] 
            del file_content
            gc.collect()
            return {"status": "success", "message": "Structural data isolated securely."}
        else:
            return {"status": "error", "message": "Unsupported file format."}
            
    except Exception as e:
        return {"status": "error", "message": f"Critical Processing Error: {str(e)}"}

# ============================================================================
# 💬 CHAT ENDPOINT (With Parallel Image Fetching)
# ============================================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[Dict[str, str]]
    use_web: bool
    temperature: float

async def fetch_pinecone(query_vector):
    if not pinecone_index: return ""
    try:
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

async def fetch_web_and_images(query, use_web):
    if not use_web: return ""
    try:
        # Fetch web text and exactly 1 image URL simultaneously
        text_results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=2)))
        image_results = await asyncio.to_thread(lambda: list(DDGS().images(query, max_results=1)))
        
        context = ""
        if text_results:
            context += "\n--- LIVE WEB DATA ---\n" + "\n".join([f"- {res['title']}: {res['body']}" for res in text_results])
        if image_results:
            context += f"\n\n[SYSTEM RESOURCE: DIAGRAM AVAILABLE AT URL: {image_results[0]['image']}]"
        return context
    except:
        # If Render IP is blocked by DuckDuckGo, fail silently without breaking the app
        return ""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        user_query = req.message.strip()
        query_vector = global_embedder.embed_query(user_query)

        user_data = active_sessions.get(req.session_id, {"vector_db": None, "cif_data": ""})
        v_db = user_data["vector_db"]
        cif_mem = user_data["cif_data"]

        # Run all three retrievers in parallel
        global_task = fetch_pinecone(query_vector)
        local_task = fetch_faiss(v_db, user_query)
        web_task = fetch_web_and_images(user_query, req.use_web)

        global_context, local_rag, web_context = await asyncio.gather(global_task, local_task, web_task)

        # Dynamic Master Prompt engineered for formatting and diagrams
        system_prompt = """You are NovaRAG, an elite Enterprise AI for Material Science engineered by Jee____ Rana Prathap Reddy.

        CRITICAL EXECUTION PROTOCOL:
        1. CONVERSATION DETECTOR: If the user query is a simple greeting (e.g., "hi", "hello"), respond naturally and briefly.
        2. DATA HIERARCHY: Synthesize your answer using the provided context in this strict order:
           [Tier 1] LOCAL WORKSPACE (Confidential user uploads. Absolute Truth.)
           [Tier 2] GLOBAL KNOWLEDGE BASE (Verified Pinecone facts.)
           [Tier 3] LIVE WEB DATA (If enabled.)
        3. INTEGRITY: Do not hallucinate numbers or chemical structures. If it is not in the context, say so.
        4. TABLES: If extracting parameters or displaying numerical data, you MUST use Markdown tables.
        5. DIAGRAMS: If the context provides a [SYSTEM RESOURCE: DIAGRAM AVAILABLE AT URL: ...], you MUST embed it in your response using Markdown image syntax: `![Diagram Description](URL_HERE)`."""

        if global_context.strip():
            system_prompt += f"\n\n--- [Tier 2] GLOBAL KNOWLEDGE BASE ---\n{global_context}"
        if local_rag.strip():
            system_prompt += f"\n\n--- [Tier 1] LOCAL WORKSPACE (CONFIDENTIAL PDF) ---\n{local_rag}"
        if cif_mem.strip():
            system_prompt += f"\n\n--- [Tier 1] LOCAL WORKSPACE (CIF DATA) ---\n{cif_mem}"
        if web_context.strip():
            system_prompt += f"{web_context}"

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
                    await asyncio.sleep(0.0001) 
            yield "data: [DONE]\n\n"
                
        return StreamingResponse(response_generator(), media_type="text/event-stream")
        
    except Exception as e:
        safe_error = f"**[System Alert]** Graceful Degradation active. Error: {str(e)}".replace("\n", "<br>")
        async def error_generator(): yield f"data: {safe_error}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")