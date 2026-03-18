# enterprise_sop_ai_traniner

This is a local, multimodal Retrieval-Augmented Generation (RAG) system built with **LlamaIndex**, **Streamlit**, and **Ollama**. It allows administrators to upload PDF Standard Operating Procedures (SOPs) and users to chat exclusively with individual documents to get accurate answers without hallucinations linking to outside files.

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed locally (to run the AI models locally on your machine).

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the necessary Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Pull the required models via Ollama. Open your terminal and run:
   ```bash
   ollama pull mxbai-embed-large
   ollama pull llava
   ollama pull llama3
   ```
   *(Note: You can tweak `VISION_LLM_MODEL` or `TEXT_LLM_MODEL` in `multimodal_rag.py` if you prefer lighter/faster variants like `llama3:8b` or `phi3`)*

## Running the Application

This is a multipage Streamlit application. To run the app, type the following in your terminal:

```bash
streamlit run app.py
```

### 1. Admin Panel
- **Access:** In the Streamlit app, expand the left sidebar and click "admin".
- **Usage:** This is where you upload your PDF SOPs. Click "Browse files" or drag-and-drop your PDFs, then click **Process Documents**. The files will be saved in the `./data` folder and ingested via `PyMuPDF` into a persistent `ChromaDB` storage locally (`./chroma_db`).
- **Management:** You can also clear the entire database from this panel if you need a fresh start.

### 2. User Interface (App)
- **Access:** Click "app" on the left sidebar pages list.
- **Usage:** This is the main chatting interface. Select a specific PDF file from the dropdown. Once selected, chat histories are kept completely separate for that file. Ask a question, and the `ExactMatchFilter` will ensure the LLM exclusively searches the chunks from the selected document!

## Notes
- To prevent database corruption, avoid deleting `./chroma_db` manually while Streamlit is processing documents. Ensure all background python tasks are killed before forceful removal.
- **Voice Module:** The application uses **Suno Bark** to generate expressive TTS. The first time you ask a question, Bark will download several gigabytes of PyTorch weights locally to your `~/.cache` directory.
- **GPU Acceleration:** The `voice_module.py` script is optimized to run Bark on CUDA (specifically targeting RTX 4060 GPUs) using the `SUNO_USE_SMALL_MODELS` hardware flag to reduce latency.
