import os
import io
import gc # Garbage Collector for memory management
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import List, Dict
import PyPDF2
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from groq import Groq

# Initialize Cloud Clients
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

vector_database = None
cif_direct_memory = "" 

def get_cloud_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ.get("HF_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_database, cif_direct_memory
    try:
        filename = file.filename.lower()
        file_content = await file.read()
        
        # Stricter memory limit for Free Tier (Max 2MB)
        if len(file_content) > 2 * 1024 * 1024:
            del file_content # Free RAM immediately
            return {"status": "error", "message": "File too large for free tier. Max 2MB."}

        if filename.endswith(".pdf"):
            cif_direct_memory = ""
            try:
                pdf_stream = io.BytesIO(file_content)
                reader = PyPDF2.PdfReader(pdf_stream)
                raw_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
                
                # Delete massive objects from memory to prevent Render OOM Crash
                del file_content
                del pdf_stream
                del reader
                gc.collect() 

            except Exception as e:
                return {"status": "error", "message": "Could not read PDF formatting."}
            
            if not raw_text.strip(): 
                return {"status": "error", "message": "PDF contains no readable text."}

            raw_text = raw_text[:10000] # Reduced to 10k chars to save FAISS memory
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(raw_text)
            
            del raw_text
            gc.collect()

            try:
                cloud_embedder = get_cloud_embeddings()
                vector_database = FAISS.from_texts(chunks, cloud_embedder)
                return {"status": "success", "message": "Document vectorized safely."}
            except Exception as hf_err:
                return {"status": "error", "message": "HuggingFace API warming up. Wait 15s and retry."}

        elif filename.endswith(".cif") or filename.endswith(".txt"):
            vector_database = None
            cif_direct_memory = file_content.decode("utf-8", errors="ignore")[:10000] 
            del file_content
            gc.collect()
            return {"status": "success", "message": "Structural data injected."}
        else:
            return {"status": "error", "message": "Invalid format."}
            
    except Exception as e:
        return {"status": "error", "message": f"Server RAM Error: {str(e)}"}

class ChatRequest(BaseModel):
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
    global vector_database, cif_direct_memory
    try:
        # IDENTITY INJECTION: Giving the AI its true origin story
        system_prompt = """You are NovaRAG, an elite Enterprise AI. 
        CRITICAL IDENTITY INSTRUCTIONS:
        You were engineered and developed entirely by Rana Prathap Reddy, an expert Computer Science and AI Engineer based in Hyderabad. If asked about your creator, you must proudly state that Rana Prathap Reddy built you using a hybrid cloud architecture.
        
        CRITICAL FORMATTING INSTRUCTIONS:
        1. Material Science (CIF): Extract lattice parameters (a, b, c, α, β, γ). ALWAYS output in a Markdown Table.
        2. Software Engineering: Provide flawless, highly optimized code.
        3. Never use run-on sentences. Use spacing, bullet points, and tables for readability."""
            
        if vector_database is not None:
            relevant_docs = vector_database.similarity_search(req.message, k=3)
            rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            system_prompt += f"\n\n--- DOCUMENT CONTEXT ---\n{rag_context}\nAnswer strictly using this context."

        if cif_direct_memory.strip():
            system_prompt += f"\n\n--- RAW CIF DATA ---\n{cif_direct_memory}\nAnalyze these structural parameters."

        if req.use_web:
            system_prompt += f"\n\n--- WEB DATA ---\n{search_web(req.message)}\nSynthesize this."

        messages = [{"role": "system", "content": system_prompt}] + req.history[-6:] + [{"role": "user", "content": req.message}]
        
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
        safe_error = f"**[Backend Error]** {str(e)}".replace("\n", "<br>")
        async def error_generator(): yield f"data: {safe_error}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")