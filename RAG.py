import streamlit as st
from ollama import chat
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set page title and add reset button in same line
col1, col2 = st.columns([4,1])
with col1:
    st.title("Chatbot with DeepSeek")
with col2:
    if st.button("New Chat"):
        # Reset to initial state with only system message
        st.session_state.messages = []       


def get_ai_response(query):
    relevant_docs = faiss_index.similarity_search(query, k=5)
    
    # Display retrieved docs
    st.markdown("### Retrieved Documents")
    for i, doc in enumerate(relevant_docs, start=1):
        st.markdown(f"**Document {i}:**\n{doc.page_content}")
    
    ctx = "\n".join([doc.page_content for doc in relevant_docs])
    query = f"Context: {ctx}\n\n Question: {query}"
    st.session_state.messages.append({'role': 'user', 'content': query})
    
    # Get response using chat
    response = chat(
        model='deepseek-r1:32b',
        messages=st.session_state.messages
    )
    
    # Add assistant response to history
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': response.message.content
    })
    
    return response.message.content

# Display chat history
for message in st.session_state.messages[1:]:  # Skip system message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add and display user message
    st.chat_message("user").markdown(prompt)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(prompt)
            st.markdown(response)