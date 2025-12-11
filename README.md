# Enterprise-Policy-RAG
A secure, local-first Enterprise RAG Dashboard for querying internal policy documents. Features hybrid OCR for legacy PDFs, vector search via ChromaDB, and offline inference using Ollama (Phi-3.5) to ensure strict data privacy. Built with Streamlit &amp; LangChain.
# Enterprise Policy RAG Assistant

A secure, local-first AI dashboard designed for internal document analysis. This tool leverages Retrieval-Augmented Generation (RAG) and Optical Character Recognition (OCR) to allow employees to query complex policy manuals, circulars, and official amendments instantly.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Stack-Streamlit%20|%20LangChain%20|%20Ollama-purple)

## 🚀 Key Features

* **RAG Architecture:** Uses Vector Search (ChromaDB) to retrieve exact policy context before generating answers.
* **Hybrid OCR Engine:** Automatically detects scanned PDFs and switches to Tesseract OCR for text extraction.
* **Secure & Local:** Runs entirely on local infrastructure using Ollama (Phi-3.5) to ensure data privacy.
* **Corporate UI:** A strict, professional dark-mode interface tailored for enterprise environments.

## 🛠️ Prerequisites

1.  **Python 3.10+**
2.  **Tesseract OCR** installed on the host machine.
3.  **Ollama** running locally with the `phi3.5` model pulled (`ollama pull phi3.5`).

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/enterprise-policy-rag.git](https://github.com/your-username/enterprise-policy-rag.git)
    cd enterprise-policy-rag
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Paths**
    Update the `tesseract_cmd` path in `src/app.py` to match your system's installation:
    ```python
    # Example for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

4.  **Run the Application**
    ```bash
    streamlit run src/app.py
    ```

## 🛡️ License

Private Enterprise Software - Internal Use Only.
