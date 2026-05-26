# -MedInsight-RAG-Assistant
Medical RAG Assistant using Streamlit, FAISS, Hugging Face embeddings, and Ollama LLMs to enable semantic retrieval, AI-powered summarization, and question answering from medical reports and clinical documents.

#  Features

-  Upload medical PDFs and images
-  OCR support for scanned reports
-  Semantic retrieval using FAISS
-  AI-powered medical question answering
-  Automatic report summarization
-  Interactive medical chat interface
-  Streamlit web application
-  Fully offline using Ollama local LLMs

---

#  Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Frontend UI |
| Ollama | Local LLM inference |
| phi3:mini | Medical response generation |
| Sentence Transformers | Text embeddings |
| FAISS | Vector database |
| Tesseract OCR | Image text extraction |
| PyPDF | PDF parsing |

---

#  System Architecture

```text
Medical Report
        ↓
Text Extraction (PDF/OCR)
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS Vector Store
        ↓
Semantic Retrieval
        ↓
LLM Context Injection
        ↓
Medical Answer Generation
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/medical-rag.git
cd medical-rag
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download:
https://ollama.com

Pull model:

```bash
ollama pull phi3:mini
```

Start Ollama:

```bash
ollama serve
```

---

## 5. Install Tesseract OCR

Download:
https://github.com/UB-Mannheim/tesseract/wiki

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 💬 Example Questions

- What abnormalities are present?
- Is creatinine level normal?
- Summarize this report
- What does elevated bilirubin indicate?
- Which values are outside normal range?

---

# 📸Features Demonstrated

Medical OCR  
Semantic Search  
Retrieval-Augmented Generation  
Local LLM Inference  
AI Summarization  
Vector Database Retrieval  

---

#  Future Improvements

- Multi-document RAG
- Chat memory
- Cloud deployment
- Medical entity highlighting
- Advanced retrieval pipelines

---

#  Disclaimer

This project is for educational and research purposes only and should not replace professional medical advice.

