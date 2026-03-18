import streamlit as st
import os
from pathlib import Path
from multimodal_rag import run_reflective_query, setup_query_engine, DATA_DIR, CHROMA_PATH
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

st.set_page_config(page_title="SAP SOP AI Trainer", page_icon="🎓", layout="wide")

st.title("🎓 SAP SOP AI Trainer (Multimodal)")
st.markdown("Select an SOP from the sidebar and ask questions. The AI will consult only the file you've selected to guide you.")

# --- Custom CSS for Buttons ---
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #28a745;
        color: white;
        border-color: #28a745;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838;
        color: white;
        border-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

def initialize_engine():
    if os.path.exists(CHROMA_PATH):
        with st.spinner("Loading AI Engine..."):
            engine = setup_query_engine()
            st.session_state.query_engine = engine

if st.session_state.query_engine is None and os.path.exists(CHROMA_PATH):
    initialize_engine()

# --- Sidebar: File Selection ---
with st.sidebar:
    st.header("Select a File to Chat")
    
    all_files = []
    if os.path.exists(DATA_DIR):
        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        
    if not all_files:
        st.info("No documents are currently available. Please ask an Administrator to upload some.")
        selected_file = None
    else:
        # Give user a radio or selectbox
        selected_file = st.selectbox("Choose a specific file:", options=all_files)
        
        # We store chat history independently per file so context doesn't mix
        if selected_file:
            chat_key = f"messages_{selected_file}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = [
                    {"role": "assistant", "content": f"You are now chatting with the data from **{selected_file}**. How can I assist you with this file?"}
                ]
            
            if st.button("Clear Chat History"):
                 st.session_state[chat_key] = [
                    {"role": "assistant", "content": f"You are now chatting with the data from **{selected_file}**. How can I assist you with this file?"}
                 ]
                 st.rerun()

# --- Main Chat Interface ---
if selected_file:
    chat_key = f"messages_{selected_file}"
    current_messages = st.session_state[chat_key]
    
    # Display the current file's chat history
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg and msg["images"]:
                for img_path in msg["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption="Context Image", use_container_width=True)
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")
            if "audio_path" in msg and msg["audio_path"] and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"], format="audio/wav")

    # Ask questions
    if prompt := st.chat_input(f"Ask a question regarding {selected_file}..."):
        if st.session_state.query_engine is None:
             st.error("The AI engine is currently not loaded. Documents may need to be uploaded by an Admin.")
        else:
             # Append user prompt immediately
             st.session_state[chat_key].append({"role": "user", "content": prompt})
             with st.chat_message("user"):
                 st.markdown(prompt)
                 
             with st.chat_message("assistant"):
                 try:
                     # Create the ExactMatchFilter so LlamaIndex restricts search to ONLY the selected file
                     filters = MetadataFilters(
                         filters=[ExactMatchFilter(key="file_name", value=selected_file)]
                     )
                     
                     dynamic_top_k = 5 # Can keep this fixed since we are only querying one file
                     
                     with st.status("Searching document...", expanded=False) as status:
                         response_stream = run_reflective_query(
                             prompt, 
                             filters=filters, 
                             top_k=dynamic_top_k
                         )
                         status.update(label="Response generated!", state="complete", expanded=False)
                     
                     # Render stream
                     if isinstance(response_stream, str):
                         full_response = response_stream
                         st.markdown(full_response)
                     else:
                         full_response = ""
                         message_placeholder = st.empty()
                         for token in response_stream.response_gen:
                             full_response += token
                             message_placeholder.markdown(full_response + "▌")
                         message_placeholder.markdown(full_response)
                         
                     # Extract sources and images
                     retrieved_text_sources = []
                     retrieved_images = []
                     if not isinstance(response_stream, str):
                         if hasattr(response_stream, "source_nodes") and response_stream.source_nodes:
                             for snode in response_stream.source_nodes:
                                 if hasattr(snode, "node"):
                                     if hasattr(snode.node, "image_path") and hasattr(snode.node, "metadata"):
                                         img_path = snode.node.metadata.get("file_path", None)
                                         if not img_path: img_path = snode.node.image_path
                                         if img_path and img_path not in retrieved_images:
                                             retrieved_images.append(img_path)
                                             
                                     if hasattr(snode.node, "metadata"):
                                         fname = snode.node.metadata.get("file_name", "Unknown")
                                         pnum = snode.node.metadata.get("page", "?")
                                         source_str = f"{fname} (page {pnum})"
                                         if source_str not in retrieved_text_sources:
                                             retrieved_text_sources.append(source_str)
                                             
                     # Display Images immediately
                     if retrieved_images:
                         for img_path in retrieved_images:
                             if os.path.exists(img_path):
                                 st.image(img_path, caption="Context Image", use_container_width=True)
                     
                     if retrieved_text_sources:
                         with st.expander("View Sources"):
                             for src in retrieved_text_sources:
                                 st.write(f"- {src}")
                                 
                     # Generate voice for the response
                     final_audio_path = None
                     try:
                         from voice_module import generate_voice
                         with st.spinner("Generating expressive audio..."):
                             audio_filename = f"response_{len(st.session_state[chat_key])}.wav"
                             final_audio_path = generate_voice(full_response, audio_filename)
                             
                             if final_audio_path and os.path.exists(final_audio_path):
                                 st.audio(final_audio_path, format="audio/wav")
                     except ImportError:
                         st.warning("Voice module not active. Please ensure Bark is installed.")
                     except Exception as e:
                         st.error(f"Voice generation failed: {e}")
                                 
                     st.session_state[chat_key].append({
                         "role": "assistant", 
                         "content": full_response,
                         "sources": retrieved_text_sources,
                         "images": retrieved_images,
                         "audio_path": final_audio_path
                     })
                         
                 except Exception as e:
                     st.error(f"An error occurred: {e}")
else:
    # If no file is selected (or none are available)
    if not all_files:
        st.warning("No files are uploaded. Please visit the Admin panel to upload documents.")
    else:
        st.info("Please select a file from the sidebar to start chatting.")
