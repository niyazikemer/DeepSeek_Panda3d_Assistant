import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import ollama



def rag_pipeline(query):
    """
    RAG Pipeline:
    1. Retrieve relevant documents using FAISS.
    2. Combine the documents with the query to create a prompt.
    3. Generate a response using the DeepSeek model via Ollama.
    """

    
    
    
    prompt = f"Question: {query}\n\nAnswer:"
    
    # Step 3: Generate response using DeepSeek via Ollama
    response = ollama.generate(model="deepseek-r1:32b", prompt=prompt)
    return response["response"]

# Streamlit UI
st.title("RAG System with DeepSeek")
st.write("Ask me anything!")

# Input box for user query
user_query = st.text_input("Enter your question:")

# Generate and display the answer
if user_query:
    with st.spinner("Generating answer..."):  # Show a spinner while processing
        answer = rag_pipeline(user_query)
    st.success("Answer:")
    st.write(answer)