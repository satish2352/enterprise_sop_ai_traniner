import os
import chromadb
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer

# ─── Custom Prompt Template ──────────────────────────────────────────────────
QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. \n"
    "IMPORTANT: \n"
    "1. Do NOT include any citations or mention file names/page numbers in your response text. \n"
    "2. If multiple files are provided in the context, treat them as independent sources unless they explicitly refer to each other. \n"
    "3. If different files provide unrelated or conflicting information, state this clearly in your response. \n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

# ─── Configuration ────────────────────────────────────────────────────────────
# Ensure you have pulled these models via ollama:
#   ollama pull mxbai-embed-large
#   ollama pull llava  (or bakllava)
#   ollama pull llama3

EMBEDDING_MODEL = "mxbai-embed-large" 
VISION_LLM_MODEL = "llava"           
TEXT_LLM_MODEL = "llama3"
CHROMA_PATH = "./chroma_db"          
DATA_DIR = "./data"                  

# ─── 1. Setup Global Settings ──────────────────────────────────────────────────
# Set the global embedding model for text
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, additional_kwargs={"num_ctx": 8192})
Settings.context_window = 8192 # Set global context window limit
# Set the global text LLM (overrides OpenAI default)
Settings.llm = Ollama(model=TEXT_LLM_MODEL, request_timeout=600.0, additional_kwargs={"num_ctx": 8192})

# ─── 2. Setup Vector Store (Chroma) ───────────────────────────────────────────
# We use ChromaDB to store text and images in distinct collections
db = chromadb.PersistentClient(path=CHROMA_PATH)

text_collection = db.get_or_create_collection("text_collection")
image_collection = db.get_or_create_collection("image_collection")

# Create distinct stores for text and images
text_store = ChromaVectorStore(chroma_collection=text_collection)
image_store = ChromaVectorStore(chroma_collection=image_collection)

# Bundle them into a StorageContext
storage_context = StorageContext.from_defaults(
    vector_store=text_store,
    image_store=image_store
)

from llama_index.core import Document
from llama_index.core.schema import ImageDocument
import fitz  # PyMuPDF

# ─── 3. Document Loading & Image Extraction ────────────────────────────────────
def ingest_documents(data_dir: str = DATA_DIR):
    """
    Parses PDFs in the directory natively using PyMuPDF to extract both text and images.
    Returns the created MultiModalVectorStoreIndex.
    """
    print(f"Reading PDFs from {data_dir}...")
    
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}. Please place your PDFs there.")
        return None

    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if files:
        # Clear existing collections to ensure exclusive upload
        print("Clearing existing collections for fresh ingestion...")
        try:
            db.delete_collection("text_collection")
            db.delete_collection("image_collection")
        except Exception:
            pass # Collections might not exist yet
            
        t_coll = db.get_or_create_collection("text_collection")
        i_coll = db.get_or_create_collection("image_collection")
        
        # Fresh stores and context to avoid stale references
        _text_store = ChromaVectorStore(chroma_collection=t_coll)
        _image_store = ChromaVectorStore(chroma_collection=i_coll)
        local_storage_context = StorageContext.from_defaults(
            vector_store=_text_store,
            image_store=_image_store
        )
    else:
        # Fallback if no files, but we usually have files if we are here
        local_storage_context = storage_context

    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Extracting text and images from: {file_path}")
        
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                documents.append(Document(text=text, metadata={"file_name": file, "page": page_num}))
                
            # Image extraction: Extract raw image bytes and save to data dir
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                img_name = f"{file}_p{page_num}_i{img_index}.{ext}"
                img_path = os.path.join(data_dir, img_name)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                    
                documents.append(ImageDocument(image_path=img_path, metadata={"file_name": file, "page": page_num, "file_path": img_path}))

    print(f"Found {len(documents)} nodes (text chunks and images). Indexing...")

    # Create the index
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=local_storage_context,
    )
    print("Indexing complete!")
    return index

# ─── 4. Query Engine Setup ─────────────────────────────────────────────────────
def setup_query_engine(filters=None, top_k=3):
    """
    Loads the existing index and sets up the Multimodal Query Engine.
    Filters can be passed to restrict the search to specific files.
    top_k determines how many text chunks to retrieve.
    """
    # Load index from the pre-populated storage context
    # Use get_or_create_collection to avoid potential errors if one was deleted
    text_collection = db.get_or_create_collection("text_collection")
    image_collection = db.get_or_create_collection("image_collection")
    
    _text_store = ChromaVectorStore(chroma_collection=text_collection)
    _image_store = ChromaVectorStore(chroma_collection=image_collection)

    index = MultiModalVectorStoreIndex.from_vector_store(
        vector_store=_text_store,
        image_vector_store=_image_store,
    )
    
    # Initialize the local Vision LLM using Ollama
    # Note: LLaVA is a 7B parameter multimodal model. Ensure Ollama is running.
    ollama_multi_modal_llm = OllamaMultiModal(
        model=VISION_LLM_MODEL,
        temperature=0.7,
        request_timeout=600.0,
        additional_kwargs={"num_ctx": 8192}
    )
    
    # Assemble the query engine
    query_engine = index.as_query_engine(
        llm=ollama_multi_modal_llm,
        text_qa_template=QA_PROMPT,
        similarity_top_k=top_k,
        image_similarity_top_k=1,
        filters=filters,
        streaming=True,
    )
    return query_engine
    

# ─── 5. Self-Reflective / Corrective Logic ────────────────────────────────────
def evaluate_relevance(query: str, nodes: list[NodeWithScore], llm) -> list[NodeWithScore]:
    """
    Evaluates the relevance of each node to the query using the LLM.
    Returns only the nodes that are deemed 'RELEVANT'.
    """
    relevant_nodes = []
    
    # We use a very concise prompt to keep it fast
    eval_prompt = (
        "TASK: Evaluate if the provided text chunk is relevant to answering the user query.\n"
        "QUERY: {query}\n"
        "TEXT CHUNK: {text_chunk}\n"
        "Your response must be exactly one word: 'RELEVANT' or 'IRRELEVANT'."
    )
    
    for node in nodes:
        text = node.node.get_content()
        response = llm.complete(eval_prompt.format(query=query, text_chunk=text[:2000]))
        if "RELEVANT" in response.text.upper():
            relevant_nodes.append(node)
            
    return relevant_nodes

def run_reflective_query(query_str: str, filters=None, top_k=5):
    """
    A manual query flow that implements 'Lite' Corrective RAG:
    1. Retrieve nodes.
    2. Evaluate relevance.
    3. Synthesize answer from relevant nodes only.
    """
    # 1. Setup Index and LLM
    text_collection = db.get_or_create_collection("text_collection")
    image_collection = db.get_or_create_collection("image_collection")
    _text_store = ChromaVectorStore(chroma_collection=text_collection)
    _image_store = ChromaVectorStore(chroma_collection=image_collection)
    
    index = MultiModalVectorStoreIndex.from_vector_store(
        vector_store=_text_store,
        image_vector_store=_image_store,
    )
    
    llm = Ollama(
        model=TEXT_LLM_MODEL, 
        temperature=0.1, # Lower temperature for evaluation
        request_timeout=300.0,
        additional_kwargs={"num_ctx": 8192}
    )
    
    # 2. Retrieve
    # LlamaIndex Multimodal retrieval setup
    retriever = index.as_retriever(similarity_top_k=top_k, image_similarity_top_k=2, filters=filters)
    nodes = retriever.retrieve(query_str)
    
    # Debugging: Check if nodes are retrieved
    print(f"DEBUG: Retrieved {len(nodes)} total semantic nodes for query '{query_str}' with filters {filters}")
    
    # 3. Evaluate (Self-Reflect)
    # We use the text LLM for fast text relevance checking
    relevant_nodes = evaluate_relevance(query_str, nodes, llm)
    
    # Ensure any retrieved Image nodes are always passed to synthesis, as our text LLM can't evaluate them
    # and they were retrieved via dedicated image embedding similarity anyway.
    for n in nodes:
        if hasattr(n.node, "image_path") and n not in relevant_nodes:
            relevant_nodes.append(n)
            
    # 4. Synthesize
    if not relevant_nodes:
        return "I'm sorry, but I couldn't find any relevant information in the selected documents to answer your question accurately."
        
    # Multimodal Synthesis LLM
    synthesis_llm = OllamaMultiModal(
        model=VISION_LLM_MODEL, 
        temperature=0.7, 
        request_timeout=600.0,
        additional_kwargs={"num_ctx": 8192}
    )
    
    # Manually format context string based on valid text chunks
    context_str = "\\n\\n".join([n.node.get_content() for n in relevant_nodes if not hasattr(n.node, "image_path")])
    image_docs = [n.node for n in relevant_nodes if hasattr(n.node, "image_path")]
    
    prompt = QA_PROMPT.format(context_str=context_str, query_str=query_str)
    
    # Stream response directly through multimodal model
    try:
        response_stream = synthesis_llm.stream_complete(prompt, image_documents=image_docs)
    except Exception as e:
        print(f"DEBUG Error in stream_complete: {e}")
        raise e
    
    class CustomStreamingResponse:
        def __init__(self, stream, nodes):
            self.response_gen = (r.delta for r in stream)
            self.source_nodes = nodes
            
    return CustomStreamingResponse(response_stream, relevant_nodes)

# ─── Execution ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Create the data folder if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Please put your SOP PDFs inside the '{DATA_DIR}' folder and rerun the script.")
        sys.exit(0)
    
    # Check if Chroma already has data, if not, ingest
    if not os.path.exists(CHROMA_PATH):
        print("No existing Qdrant database found. Starting ingestion process...")
        index = ingest_documents(DATA_DIR)
        if index is None:
            sys.exit(0)
    else:
        print("Existing Qdrant database found. Skipping ingestion...")

    print("\nSetting up the query engine...")
    query_engine = setup_query_engine()
    
    # Execute query
    print("\n=======================================================")
    print("Multimodal RAG Pipeline Ready (LlamaIndex + Ollama)")
    print("=======================================================\n")
    
    question = " ".join(sys.argv[1:])
    if not question:
        question = "What is the next step after defect recording?"
        
    print(f"Question: {question}")
    print("Thinking...")
    try:
        response = query_engine.query(question)
        print("\n---------- ANSWER ----------")
        print(response.response)
        
        # Print sources for verification
        print("\n[Sources Retrieved]")
        for node in response.metadata.get("image_nodes", []):
            print(f"- Image: {node.metadata.get('file_name', node.id_)}")
        for node in response.metadata.get("text_nodes", []):
            print(f"- Text Fragment from: {node.metadata.get('file_name', 'Unknown')}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error querying: {e}")
        
    # Explicitly close the database client to prevent late-stage Python ImportError
    # Chroma PersistentClient does not have a strict close() method natively like Qdrant does,
    # but we can just pass.
    pass
