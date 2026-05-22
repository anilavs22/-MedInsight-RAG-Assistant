
# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import tempfile
import hashlib
import numpy as np
import faiss
import ollama

from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import pytesseract

# =====================================================
# TESSERACT PATH
# =====================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="MedInsight - RAG Assistant",

    page_icon="🩺",

    layout="centered"
)

# =====================================================
# LIGHT PINK UI
# =====================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #ffe4ec;
    color: black;
}

h1,h2,h3,h4,h5,h6,p,div,label,span {
    color: black !important;
}

[data-testid="stFileUploader"] {
    background-color: #ffd6e7;
    border-radius: 12px;
    padding: 10px;
}

.stTextInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}

.stButton > button {
    background-color: #ff69b4;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #ff1493;
    color: white;
}

.response-box {
    background-color: #fff0f5;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ffb6c1;
    margin-top: 10px;
    color: black !important;
}

.response-box * {
    color: black !important;
}

</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =====================================================
# SESSION STATE
# =====================================================

if "summary" not in st.session_state:

    st.session_state.summary = None

if "last_hash" not in st.session_state:

    st.session_state.last_hash = ""

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted + "\n"

    return text

# =====================================================
# IMAGE OCR
# =====================================================

def extract_text_from_image(image):

    text = pytesseract.image_to_string(image)

    # basic cleanup
    text = text.replace("\n\n", "\n")
    text = text.replace("|", "")
    text = text.replace("mgidL", "mg/dL")

    return text

# =====================================================
# TEXT SPLITTING
# =====================================================

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=400,

        chunk_overlap=50
    )

    return splitter.split_text(text)

# =====================================================
# VECTOR STORE
# =====================================================

@st.cache_resource
def create_vector_store(content_hash, chunks_tuple):

    chunks = list(chunks_tuple)

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index, chunks

# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_relevant_chunk(

    query,

    chunks,

    index
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        1
    )

    idx = indices[0][0]

    return chunks[idx]

# =====================================================
# TITLE
# =====================================================

st.title("🩺 MedInsight - RAG Assistant")

st.caption(
    "Upload medical report and ask questions"
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(

    "Upload Medical Report",

    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)

# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file is not None:

    suffix = uploaded_file.name.split(".")[-1]

    file_bytes = uploaded_file.read()

    content_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]

    # reset summary for new file
    if content_hash != st.session_state.last_hash:

        st.session_state.summary = None

        st.session_state.last_hash = content_hash

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=f".{suffix}"

    ) as tmp:

        tmp.write(file_bytes)

        temp_path = tmp.name

    # =====================================================
    # EXTRACT TEXT
    # =====================================================

    with st.spinner("Extracting text..."):

        if suffix == "pdf":

            document_text = extract_text_from_pdf(
                temp_path
            )

        else:

            image = Image.open(temp_path)

            document_text = extract_text_from_image(
                image
            )

    if not document_text.strip():

        st.error(
            "No text extracted."
        )

        st.stop()

    st.success(
        "Medical report processed successfully."
    )

    # =====================================================
    # CHUNKING
    # =====================================================

    chunks = split_text(
        document_text
    )

    # =====================================================
    # VECTOR STORE
    # =====================================================

    with st.spinner("Building knowledge base..."):

        index, cached_chunks = create_vector_store(

            content_hash,

            tuple(chunks)
        )

   # =====================================================
# SUMMARY
# =====================================================

st.markdown("---")

st.subheader("📋 Report Summary")

if st.button("Generate Summary"):

    with st.spinner("Generating summary..."):

        try:

            response = ollama.chat(

                model="phi3:mini",

                messages=[

                    {
                        "role": "system",

                        "content":
                        """
                        Generate a professional medical summary in bullet points.

                        Rules:
                        - Use only bullet points
                        - No introduction
                        - No greetings
                        - No repeated words
                        - No incomplete sentences
                        - Mention abnormal findings first
                        - Include test name, value, status, and short interpretation
                        - Keep explanations concise but informative
                        - Maximum 10 bullet points
                        - Ignore OCR artifacts or broken symbols

                        Format:
                        • Test → Value (Status) — Clinical interpretation

                        Example:
                        • Hemoglobin → 11.1 g/dL (Low) — Mild anemia may be present.

                        Final bullet:
                        • Overall Impression → Short clinical summary
                        """
                    },

                    {
                        "role": "user",

                        "content":
                        document_text[:2500]
                    }
                ],

                stream=False,

                options={

                    "temperature": 0.2,

                    "num_predict": 300
                }
            )

            st.session_state.summary = response[
                "message"
            ]["content"]

        except Exception as e:

            st.error(str(e))

# DISPLAY SUMMARY
if st.session_state.summary is not None:

    st.markdown(

        f"""
        <div class="response-box">
        {st.session_state.summary}
        </div>
        """,

        unsafe_allow_html=True
    )

    # =====================================================
    # SIMPLE CHAT
    # =====================================================

    st.markdown("---")

    st.subheader("💬 Ask Questions")

    user_query = st.text_input(
        "Ask medical question"
    )

    if user_query:

        with st.spinner("Generating answer..."):

            try:

                context = retrieve_relevant_chunk(

                    query=user_query,

                    chunks=cached_chunks,

                    index=index
                )

                response = ollama.chat(

                    model="phi3:mini",

                    messages=[

                        {
                            "role": "system",

                            "content":
                            f"""
                            Answer ONLY from this medical report.

                            Keep answers concise and medically relevant.

                            Context:
                            {context[:800]}
                            """
                        },

                        {
                            "role": "user",

                            "content": user_query
                        }
                    ],

                    stream=False,

                    options={

                        "temperature": 0.2,

                        "num_predict": 120
                    }
                )

                answer = response[
                    "message"
                ]["content"]

                st.markdown(

                    f"""
                    <div class="response-box">
                    {answer}
                    </div>
                    """,

                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(str(e))

# =====================================================
# EMPTY STATE
# =====================================================

else:

    st.info(
        "Upload a medical report to begin."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Educational Medical AI Assistant"
)
