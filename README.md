# 🌌 NovaRAG: Material Science AI Assistant

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688)
![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000)
![Firebase](https://img.shields.io/badge/Firebase-Auth_%7C_Cloud_Sync-FFCA28)

**NovaRAG** is a smart, secure AI chat application built specifically for Material Science researchers. 

Standard AI models often "guess" or hallucinate numbers when asked about complex scientific data. NovaRAG solves this by letting users upload their own research PDFs and `.cif` (Crystallographic Information) files. The AI reads *only* your documents to give you 100% accurate structural data and lattice parameters.

🔗 **[Live Production Deployment](https://nova-rag.vercel.app)** *(Login via Google to access secure workspaces)*

---

## ✨ What It Does

* **Strictly Accurate Answers:** Upload dense scientific PDFs or `.cif` files. The AI extracts the exact numbers and formulas into clean tables without making things up.
* **Global vs. Local Knowledge:** The app uses a "Dual-Brain" approach. It has a global database filled with verified public Material Science facts, and a secure local database just for the file you uploaded.
* **Smart Data Privacy:** When you upload a confidential document, it is stored in temporary memory. As soon as you close your session, the document is completely wiped from the server. 
* **Cloud Workspaces:** Log in securely with Google. Your chat history and different project workspaces are saved and synced across all your devices.

---

## 🏗️ How It Was Built

This project is broken into two main parts to keep it fast and stable:

1. **The Frontend (User Interface):** * Built with standard HTML, JavaScript, and Tailwind CSS.
   * Hosted on Vercel. 
   * Features a dark-mode design, syntax highlighting for code, and a multi-workspace sidebar.
2. **The Backend (AI Engine):** * Built with Python and FastAPI, hosted on Render.
   * Uses **Llama-3** (via Groq) for lightning-fast text generation.
   * Uses **Pinecone** to store the permanent global Material Science knowledge.
   * Uses **FAISS** to temporarily store user-uploaded files in RAM.

---

## 💻 Run It Yourself

To run the backend server on your own computer:

```bash
# Clone the repository
git clone [https://github.com/ranaprathapreddy597/NovaRAG](https://github.com/ranaprathapreddy597/NovaRAG)
cd NovaRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install fastapi uvicorn python-multipart PyPDF2 duckduckgo-search langchain langchain-community faiss-cpu sentence-transformers groq pinecone

👨‍💻 Built By
Rana Prathap Reddy Jeedipally

Computer Science & Engineering Student

Focused on Full-Stack Development and Practical AI Solutions

📧 Email: ranaprathapreddyj@gmail.com