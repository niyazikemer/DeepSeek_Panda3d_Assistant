import streamlit as st
from ollama import chat

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'system',
            'content': """You are a helpful AI assistant. Always provide accurate, 
            informative, and engaging responses while maintaining a conversational tone."""
        }
    ]

# Set page title
st.title("Chatbot with DeepSeek")

def get_ai_response(query):
    # Add user message to history
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