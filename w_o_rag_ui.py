import streamlit as st
import ollama

# Initialize chat history and system prompt
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.system_prompt = """You are a helpful AI assistant. Always provide accurate, 
    informative, and engaging responses while maintaining a conversational tone."""
    st.session_state.last_context = None  # Store last context for reuse

# Set page title
st.title("Chatbot with DeepSeek")

def get_ai_response(query):
    # First generate with just the query to get context
    if st.session_state.last_context is None:
        initial_response = ollama.generate(
            model="deepseek-r1:32b",
            prompt=query,
            system=st.session_state.system_prompt
        )
        st.session_state.last_context = initial_response.get('context', None)
    
    # Generate final response using context
    response = ollama.generate(
        model="deepseek-r1:32b",
        prompt=query,
        system=st.session_state.system_prompt,
        context=st.session_state.last_context
    )
    
    # Update context for next iteration
    st.session_state.last_context = response.get('context', None)
    
    return response["response"]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})