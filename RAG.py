import streamlit as st
from ollama import chat
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from hybrid_retriever import HybridRetriever
from re_ranker import OptimizedReranker
from agent import Agent  # Import the Agent class

# Custom CSS to move sidebar to the right
#put the combined documnet at the right side of the screen
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            left: auto !important;
            right: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

embeddings = HuggingFaceEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and reset button
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chatbot with DeepSeek")
with col2:
    if st.button("New Chat"):
        st.session_state.messages = []

reranker = OptimizedReranker()
agent = Agent()  # Initialize the Agent

def get_ai_response(query):
    # Stage 1: Broad retrieval
    hybrid_retriever = HybridRetriever(faiss_index)
    initial_docs = hybrid_retriever.hybrid_search(query, k=100)
    
    # Stage 2: Reranking
    reranked_docs = reranker.rerank(query, initial_docs, top_k=20)
    
    # Analyze documents using the agent
    if not agent.analyze_documents(reranked_docs):
        improved_query = agent.generate_improved_query(query, reranked_docs)
        st.warning("Document analysis found insufficient information. Suggested improved query:")
        st.markdown(f"**{improved_query}**")
        return improved_query
    
    # Display in sidebar
    with st.sidebar:
        st.markdown("### Retrieved Documents")
        for i, doc in enumerate(reranked_docs, start=1):
            with st.expander(f"Document {i}"):
                # Show preview first
                st.markdown("**Preview:**")
                st.markdown(f"{doc.page_content[:200]}...")
                st.divider()
                # Show full content
                st.markdown("**Full Content:**")
                st.markdown(doc.page_content)
                st.divider()
                # Show metadata without nested expander
                st.markdown("**Metadata:**")
                st.json(doc.metadata)
    
    # Prepare context
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(reranked_docs)
    ])
    augmented_query = f"{context}\n\nQuestion: {query}"
    st.session_state.messages.append({'role': 'user', 'content': augmented_query})
    
    # Create placeholder for streaming response
    response_placeholder = st.empty()
    full_response = ""
    
    # Stream the response
    stream = chat(
        model='deepseek-r1:32b',
        messages=st.session_state.messages,
        options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50, 'num_ctx': 30000},
        stream=True
    )
    
    # Process streaming chunks
    for chunk in stream:
        if chunk.message.content:
            full_response += chunk.message.content
            response_placeholder.markdown(full_response + "▌")
    
    # Final update without cursor
    response_placeholder.markdown(full_response)
    
    # Add to chat history
    st.session_state.messages.append({
        'role': 'assistant',
        'content': full_response
    })
    
    return full_response

# Main chat area (full width)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        response = get_ai_response(prompt)