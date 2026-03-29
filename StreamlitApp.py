import streamlit as st
import os
import sys
from QA_with_pdf.data_ingestion import load_data
from QA_with_pdf.embedding import load_embedding
from QA_with_pdf.model_api import load_model
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from errorhandling import CustomException
from logger import logger

def main():
    st.set_page_config(page_title="QA with PDF", page_icon="📄", layout="wide")
    
    st.header("QA System using Gemini & LlamaIndex 📄")
    
    with st.sidebar:
        st.title("Upload PDF")
        uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
        
        if st.button("Submit & Process"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    try:
                        # Save uploaded file temporarily
                        data_dir = "data"
                        os.makedirs(data_dir, exist_ok=True)
                        filepath = os.path.join(data_dir, uploaded_file.name)
                        with open(filepath, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        logger.info(f"File saved: {filepath}")
                        
                        # Load Data
                        documents = load_data(data_dir)
                        
                        # Load Model & Embeddings
                        model = load_model()
                        embedding_model = load_embedding()
                        
                        # Configure Settings (New LlamaIndex v0.10+ syntax)
                        Settings.llm = model
                        Settings.embed_model = embedding_model
                        
                        # Create Index
                        logger.info("Creating Index...")
                        index = VectorStoreIndex.from_documents(documents)
                        
                        # Persist Index
                        index.storage_context.persist()
                        logger.info("Index created and persisted.")
                        
                        st.success("Analysis Complete! You can now ask questions.")
                        
                    except Exception as e:
                        logger.error(f"Error during processing: {str(e)}")
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please upload a PDF file first.")

    # User Query Section
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        try:
            # Check if index exists
            if os.path.exists("./storage"):
                # Load Model & Embeddings again for query context (or rely on global settings if set)
                model = load_model()
                embedding_model = load_embedding()
                Settings.llm = model
                Settings.embed_model = embedding_model

                storage_context = StorageContext.from_defaults(persist_dir="./storage")
                index = load_index_from_storage(storage_context)
                
                query_engine = index.as_query_engine()
                response = query_engine.query(user_question)
                
                st.write("### Answer:")
                st.write(response.response)
                
                with st.expander("Debug: Context retrieved from LlamaIndex"):
                    for i, n in enumerate(response.source_nodes):
                        st.write(f"**Node {i}:**")
                        st.text(n.node.get_content())
            else:
                st.error("Index not found. Please upload and process a PDF first.")
        except Exception as e:
            logger.error(f"Error during querying: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
