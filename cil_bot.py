import streamlit as st
import os
import tempfile
import time

# --- LIBRARIES ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    
    # OCR Libraries
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    st.error("System Error: Critical libraries missing. Contact IT Support.")
    st.stop()

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- PAGE CONFIGURATION ---
favicon_path = "coal-india-limited-cil-logo-png_seeklogo-386974.png"
page_icon = favicon_path if os.path.exists(favicon_path) else None 

st.set_page_config(
    page_title="CIL Policy Neural Engine",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FANG-TIER UI SYSTEM ---
st.markdown("""
<style>
    /* 1. FONT IMPORT */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* 2. CSS VARIABLES */
    :root {
        --background: #09090b;
        --sidebar-bg: #0f1115;
        --card-bg: rgba(255, 255, 255, 0.03);
        --border-color: rgba(255, 255, 255, 0.08);
        --primary-gradient: linear-gradient(90deg, #3B82F6, #6366F1);
        --text-primary: #ECECEC;
        --text-secondary: #A1A1AA;
    }

    /* 3. GLOBAL RESETS */
    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp::before {
        content: "";
        position: absolute;
        top: -10%;
        left: -10%;
        width: 120%;
        height: 120%;
        background-image: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* 4. TYPOGRAPHY & ICON PROTECTION */
    h1, h2, h3, h4, p, span, div, button {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }
    
    [data-testid="stSidebarCollapseButton"] > span, 
    [data-testid="stSidebarExpandButton"] > span {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* 5. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
        padding-top: 2rem;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        z-index: 100;
    }
    
    [data-testid="stSidebarCollapseButton"] {
        color: white !important;
    }

    /* 6. CUSTOM TITLES */
    .hero-title {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.05rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: var(--text-secondary) !important;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.02rem;
        margin-bottom: 2rem;
    }

    /* 7. CHAT INTERFACE */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    [data-testid="stChatMessage"] [data-testid="stImage"] {
        border-radius: 50%;
        border: 2px solid #3B82F6;
    }

    /* 8. INPUT FIELD */
    .stChatInputContainer textarea {
        background-color: #18181B !important;
        color: #F4F4F5 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 14px 20px;
        font-size: 0.95rem;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* 9. SKELETON LOADER (New Addition) */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .skeleton-line {
        height: 12px;
        background: linear-gradient(to right, #27272a 4%, #3f3f46 25%, #27272a 36%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite linear;
        border-radius: 4px;
        margin-bottom: 10px;
    }

    /* 10. FILE UPLOADER & BUTTONS */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px dashed var(--border-color);
        border-radius: 12px;
        padding: 20px;
    }
    
    div.stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.02rem;
        transition: all 0.2s;
        width: 100%;
        text-transform: uppercase;
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }

    /* 11. WELCOME CARD (SVG) */
    .welcome-container {
        border: 1px solid var(--border-color);
        background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.1), transparent 40%), var(--card-bg);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    .feature-box {
        background: rgba(255,255,255,0.02); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }
    
    .icon-svg {
        width: 24px;
        height: 24px;
        stroke: #3B82F6;
        stroke-width: 2;
        fill: none;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--background);
    }
    ::-webkit-scrollbar-thumb {
        background: #3F3F46;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS (UNCHANGED) ---

def extract_text_with_ocr(pdf_path, file_name):
    """Hybrid extraction: Text + OCR for scanned pages."""
    documents = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if len(text.strip()) < 50: 
                pix = page.get_pixmap(dpi=200) 
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(image)
            
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_num + 1}
                ))
        doc.close()
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return []
    return documents

def build_vector_db(uploaded_files):
    """Processes files and builds Vector DB."""
    all_docs = []
    progress_text = "Processing..."
    my_bar = st.progress(0, text=progress_text)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        my_bar.progress((i / total_files), text=f"Analyzing {uploaded_file.name} ({i+1}/{total_files})")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            
        file_docs = extract_text_with_ocr(tmp_path, uploaded_file.name)
        all_docs.extend(file_docs)
        try:
            os.remove(tmp_path)
        except:
            pass

    my_bar.progress(1.0, text="Indexing Knowledge Base...")
    time.sleep(0.5)
    my_bar.empty()
    
    if not all_docs:
        return None

    # Split & Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    collection_name = f"cil_kb_{int(time.time())}"
    return Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=collection_name)

def get_answer_chain(vector_db):
    """RAG Chain Setup."""
    llm = Ollama(model="phi3.5")
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    template = """You are an internal AI Policy Analyst for Coal India Limited.
    Strictly use the provided context to answer the employee's query professionally and concisely.
    If the answer is not explicitly stated in the documents, state: "Reference not found in uploaded policies."
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- UI LAYOUT ---

logo_file = "coal-india-limited-cil-logo-png_seeklogo-386974.png"

# 1. HEADER
col1, col2 = st.columns([0.8, 8])
with col1:
    if os.path.exists(logo_file):
        st.image(logo_file, width=70)
    else:
        st.markdown("<div style='font-size: 2rem; font-weight: 700; color: #3B82F6;'>CIL</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="hero-title">COAL INDIA LIMITED</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">INTELLIGENT POLICY ANALYTICS ENGINE</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.1); margin: 20px 0;'></div>", unsafe_allow_html=True)

# 2. SIDEBAR
with st.sidebar:
    st.markdown("### Data Ingestion")
    st.caption("Upload restricted policy manuals.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        if num_files > 10:
             st.error("Limit: 10 files.")
        else:
            st.success(f"{num_files} Files Ready")
            current_file_names = ",".join(sorted([f.name for f in uploaded_files]))
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            if "last_processed_bundle" not in st.session_state or st.session_state.last_processed_bundle != current_file_names:
                if st.button("Initialize Pipeline", use_container_width=True):
                    with st.spinner("Processing Vectors..."):
                        db = build_vector_db(uploaded_files)
                        if db:
                            st.session_state.vector_db = db
                            st.session_state.last_processed_bundle = current_file_names
                            st.session_state.chat_history = [] 
                            st.success("Database Synchronized")
                        else:
                            st.error("Vectorization Failed")
            else:
                st.info("System Operational")

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("### Controls")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.chat_history = []
        if "vector_db" in st.session_state:
            del st.session_state.vector_db
        if "last_processed_bundle" in st.session_state:
             del st.session_state.last_processed_bundle
        st.rerun()
        
    st.markdown("""
    <div style='position: fixed; bottom: 20px; left: 20px; font-size: 0.75rem; color: #52525B;'>
        INTERNAL USE ONLY • v3.0.0
    </div>
    """, unsafe_allow_html=True)

# 3. CHAT WORKSPACE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    search_icon = """<svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 24 24" stroke="currentColor" fill="none"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>"""
    ocr_icon = """<svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 24 24" stroke="currentColor" fill="none"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>"""
    lock_icon = """<svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 24 24" stroke="currentColor" fill="none"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>"""

    st.markdown(f"""
    <div class="welcome-container">
        <h2 style="color: white; font-weight: 600;">Welcome to Policy Neural Engine</h2>
        <p style="color: #A1A1AA; font-size: 1.1rem; max-width: 600px; margin: 10px auto;">
            Securely query corporate policy using advanced RAG architecture.
            Data remains local and encrypted.
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 30px;">
            <div class="feature-box">
                {search_icon}
                <span style="font-weight: 500; color: #ECECEC;">Semantic Search</span>
            </div>
            <div class="feature-box">
                {ocr_icon}
                <span style="font-weight: 500; color: #ECECEC;">OCR Integration</span>
            </div>
            <div class="feature-box">
                {lock_icon}
                <span style="font-weight: 500; color: #ECECEC;">Enterprise Safe</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat
for message in st.session_state.chat_history:
    avatar_img = logo_file if message["role"] == "assistant" and os.path.exists(logo_file) else None
    with st.chat_message(message["role"], avatar=avatar_img):
        st.markdown(message["content"])

# 4. INPUT AREA
prompt = st.chat_input("Query the knowledge base...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=None):
        st.markdown(prompt)

    if "vector_db" in st.session_state and st.session_state.vector_db:
        avatar_img = logo_file if os.path.exists(logo_file) else None
        
        with st.chat_message("assistant", avatar=avatar_img):
            # NEW: Skeleton Loader instead of simple spinner
            placeholder = st.empty()
            placeholder.markdown("""
            <div class="skeleton-line" style="width: 100%;"></div>
            <div class="skeleton-line" style="width: 80%;"></div>
            <div class="skeleton-line" style="width: 60%;"></div>
            """, unsafe_allow_html=True)
            
            # Generate Response
            chain = get_answer_chain(st.session_state.vector_db)
            full_response = chain.invoke(prompt)
            
            # Clear skeleton and Stream response (Typewriter effect)
            placeholder.empty()
            message_placeholder = st.empty()
            displayed_response = ""
            
            # Simulating streaming for UX (since we use chain.invoke which is blocking)
            # In a real async chain, we would yield chunks. Here we simulate the effect.
            for chunk in full_response.split():
                displayed_response += chunk + " "
                message_placeholder.markdown(displayed_response + "▌")
                time.sleep(0.02) # Typing speed
            
            message_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant", avatar=avatar_img):
            error_msg = "⚠️ **System Alert:** Knowledge Base is empty. Please initialize pipeline via sidebar."
            st.markdown(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})