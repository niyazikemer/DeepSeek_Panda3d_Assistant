import streamlit as st
from ollama import chat
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from hybrid_retriever import HybridRetriever

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

def get_ai_response(query):
    
    hybrid_retriever = HybridRetriever(faiss_index)
    relevant_docs = hybrid_retriever.hybrid_search(query, k=20)
    #relevant_docs = faiss_index.similarity_search(query, k=5)
    #this is for giving the model some liberty to choose the best documents and not all of them
    context_explanation = "some of the Documents could be relevant, some of them might not be. Please use them considering my question below. Here are the top 5 documents that might help you:"
    # Display docs in right sidebar
    with st.sidebar:
        st.markdown("### Retrieved Documents")
        for i, doc in enumerate(relevant_docs, start=1):
            st.markdown(f"**Document {i}:**\n{doc.page_content}")
            st.divider()
    
    ctx = "\n".join([doc.page_content for doc in relevant_docs])
    augmented_query = f"Context:{context_explanation} {ctx}\n\nQuestion: {query}"
    st.session_state.messages.append({'role': 'user', 'content': augmented_query})
    
    response = chat(
        model='deepseek-r1:32b',
        messages=st.session_state.messages,
        #https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50,'num_ctx': 15000}
    )
    
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': response.message.content

    })
    
    return response.message.content

# Main chat area (full width)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(prompt)
            st.markdown(response)