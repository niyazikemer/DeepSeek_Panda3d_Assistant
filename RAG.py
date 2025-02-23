import streamlit as st
from ollama import chat as ollama_chat
from anthropic import Anthropic
import os
from prepend_context import GeminiEmbeddings
from langchain.vectorstores import FAISS
from hybrid_retriever import HybridRetriever
from re_ranker import OptimizedReranker
from agent import Agent  # Import the Agent class

# Custom CSS to move sidebar to the right
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            left: auto !important;
            right: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# embeddings = HuggingFaceEmbeddings()
embeddings = GeminiEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and reset button
col1, col2 = st.columns([4, 1])
with col1:
    st.title("The HH Guide")
with col2:
    if st.button("New Chat"):
        st.session_state.messages = []

reranker = OptimizedReranker()
agent = Agent()  # Initialize the Agent

# Add model selection in sidebar
with st.sidebar:
    model_choice = st.selectbox(
        "Select Model",
        ["Claude Sonnet", "Deepseek Ollama"],
        index=1  # Default to Deepseek
    )

def get_claude_response(messages, context, query):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Debug input state
    print("\n=== Claude Response Debug ===")
    print(f"Messages count: {len(messages)}")
    print(f"Context length: {len(context)}")
    print(f"Query: {query}")
    
    # System message to set context
    system_message = "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."
    
    # Convert context and query to Claude format with system message
    formatted_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the provided context."
    
    # Create placeholder for streaming response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        print("\n=== Starting Claude Stream ===")
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.7,
            system=system_message,
            messages=[
                *[{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                {"role": "user", "content": formatted_prompt}
            ]
        ) as stream:
            #print("Stream created successfully")
            for chunk in stream:
                #print(f"\n=== Chunk Type: {chunk.type} ===")
                
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    new_text = chunk.delta.text
                    #print(f"New text received: {new_text[:50]}..." if len(new_text) > 50 else new_text)
                    full_response += new_text
                    response_placeholder.markdown(full_response + "▌")
                
                elif hasattr(chunk, 'message'):
                    #print("Complete message received")
                    if hasattr(chunk.message, 'content'):
                        for block in chunk.message.content:
                            if block.type == 'text':
                                full_response += block.text
                                response_placeholder.markdown(full_response + "▌")
    
    except Exception as e:
        # print(f"\n=== Error in Claude Stream ===")
        # print(f"Error type: {type(e)}")
        # print(f"Error details: {str(e)}")
        st.error(f"Error calling Claude API: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."
    
    print(f"\n=== Final Response ===")
    print(f"Response length: {len(full_response)}")
    if full_response:
        print(f"First 200 chars: {full_response[:200]}...")
    else:
        print("No response generated")
    
    # Final update without cursor
    response_placeholder.markdown(full_response)
    return full_response

def get_ai_response(query):
    # Stage 1: Broad retrieval
    hybrid_retriever = HybridRetriever(faiss_index)
    initial_docs = hybrid_retriever.hybrid_search(query, k=100)
    
    # Stage 2: Reranking
    reranked_docs = reranker.rerank(query, initial_docs, top_k=20)
    
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

    is_enough, related_docs = agent.analyze_documents(query, reranked_docs)
    if not is_enough:
        improved_query = agent.generate_improved_query(query, reranked_docs)
        st.warning("Document analysis found insufficient information. Suggested improved query:")
        st.markdown(f"**{improved_query}**")
        print("Improved query:", improved_query)
        return improved_query
    else:
        context = related_docs

    augmented_query = f"{context}\n\nQuestion: {query}"
    st.session_state.messages.append({'role': 'user', 'content': augmented_query})
    
    if model_choice == "Claude Sonnet":
        full_response = get_claude_response(
            st.session_state.messages,
            context,
            query
        )
    else:  # Deepseek Ollama
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response using Ollama
        stream = ollama_chat(
            model='deepseek-r1:32b',
            messages=st.session_state.messages,
            options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50, 'num_ctx': 30000},
            stream=True
        )
        
        for chunk in stream:
            if chunk.message.content:
                full_response += chunk.message.content
                response_placeholder.markdown(full_response + "▌")
        
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