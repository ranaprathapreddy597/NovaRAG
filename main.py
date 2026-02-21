import os
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# LAZY LOADING: We do not download the model at startup anymore.
embed_model = None
vector_database = None
cif_direct_memory = "" 

def get_embed_model():
    global embed_model
    if embed_model is None:
        print("First PDF uploaded! Initializing Embedding Engine now...")
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embed_model

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_database, cif_direct_memory
    try:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            cif_direct_memory = ""
            try:
                reader = PyPDF2.PdfReader(file.file)
                raw_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
            except:
                return {"status": "error", "message": "Corrupted PDF."}
            
            if not raw_text.strip(): return {"status": "error", "message": "Empty PDF."}

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(raw_text)
            
            # The model is loaded here only when needed
            active_model = get_embed_model()
            vector_database = FAISS.from_texts(chunks, active_model)
            
            return {"status": "success", "message": "Document indexed."}

        elif filename.endswith(".cif") or filename.endswith(".txt"):
            vector_database = None
            raw_data = await file.read()
            cif_direct_memory = raw_data.decode("utf-8")[:10000] 
            return {"status": "success", "message": "CIF data injected."}
        else:
            return {"status": "error", "message": "Use .pdf or .cif"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]
    use_web: bool
    temperature: float

def search_web(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        return "\n".join([f"- {res['title']}: {res['body']}" for res in results]) if results else "No data."
    except:
        return "Search failed."

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    global vector_database, cif_direct_memory
    try:
        system_prompt = """You are NovaRAG, an elite Enterprise AI. 
        CRITICAL INSTRUCTIONS:
        1. When asked about Material Science (like a CIF file), extract the lattice parameters (a, b, c, α, β, γ) and atomic data.
        2. ALWAYS output this data in a clean Markdown Table. Never output raw data in a run-on sentence. Be highly detailed.
        3. For coding queries (C++, RPA), provide flawless, deeply explained code blocks.
        4. Use proper newlines for all output to ensure formatting."""
            
        if vector_database is not None:
            relevant_docs = vector_database.similarity_search(req.message, k=4)
            rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            system_prompt += f"\n\n--- DOCUMENT CONTEXT ---\n{rag_context}\nUse this context heavily."

        if cif_direct_memory.strip():
            system_prompt += f"\n\n--- RAW CIF DATA ---\n{cif_direct_memory}\nAnalyze these structural parameters thoroughly."

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