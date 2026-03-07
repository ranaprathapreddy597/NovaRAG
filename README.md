# 🌌 NovaRAG: Enterprise Context-Aware AI

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC)
![Firebase](https://img.shields.io/badge/Firebase-Auth_%7C_Firestore-FFCA28)
![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036)

**NovaRAG** is a decoupled, multi-tenant Enterprise SaaS application. It utilizes a **Retrieval-Augmented Generation (RAG)** pipeline to allow users to securely chat with their proprietary documents without risking data leakage or AI hallucination. 

Engineered specifically for complex workflows in **Material Science (CIF structure analysis)** and **Robotic Process Automation (Automation Anywhere)**.

---

## 🚀 Key Features

* **🔒 Secure Document Ingestion (RAG):** Upload dense PDFs, `.txt`, or `.cif` files (up to 15MB). The system chunks, vectorizes, and queries strictly against your document. 
* **⚡ Ultra-Low Latency Inference:** Powered by **Groq's LPUs** (Language Processing Units) and Llama-3, providing near-instant text streaming.
* **🧠 Multi-Tenant Memory Isolation:** Uses an in-memory **FAISS** vector database tied to secure session IDs. User A cannot access User B's structural data.
* **🔄 Cloud Workspace Sync:** Integrated with **Google Firebase**. Users can log in via OAuth, manage multiple smart-titled workspaces, and sync chat histories across devices.
* **📱 Progressive Web App (PWA):** Fully responsive, dark-mode user interface designed with Tailwind CSS. Installable natively on mobile devices.
* **🛡️ Immutability Guardrails:** Hard-coded system constraints prevent jailbreaks, prompt injection, and hallucinated data extraction.

---

## 🏗️ System Architecture

NovaRAG operates on a decoupled Client-Server architecture to optimize heavy machine learning workloads:

1.  **Frontend (Client Edge):** Built with Vanilla JavaScript and Tailwind CSS. Hosted on Vercel for global edge caching. Handles OAuth, markdown rendering, syntax highlighting, and state management.
2.  **Backend (Inference Server):** A Python **FastAPI** REST engine hosted on Render. Handles heavy file processing and API routing.
3.  **Data Pipeline:** * **PyPDF2** for raw text extraction.
    * **LangChain** for semantic recursive text splitting.
    * **Hugging Face (`all-MiniLM-L6-v2`)** for generating mathematical text embeddings.
4.  **Database:** **Firestore (NoSQL)** for persistent chat history and **FAISS** for transient, session-based vector storage.

---

## 💻 Local Development Setup

To run NovaRAG locally, you need to spin up both the Python backend and the web frontend.

### 1. Backend Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/NovaRAG.git](https://github.com/yourusername/NovaRAG.git)
cd NovaRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt