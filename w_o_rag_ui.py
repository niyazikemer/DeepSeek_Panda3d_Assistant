import streamlit as st
import ollama

# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set page title
st.title("Chatbot with DeepSeek")

def get_ai_response(query):
    prompt = f"Question: {query}\n\nAnswer:"
    response = ollama.generate(model="deepseek-r1:32b", prompt=prompt)
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