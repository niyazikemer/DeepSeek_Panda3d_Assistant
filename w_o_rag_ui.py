import streamlit as st
from ollama import chat

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set page title and add reset button
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chatbot with DeepSeek")
with col2:
    if st.button("New Chat"):
        st.session_state.messages = []

# Display chat history (skip system message)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to UI and history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        stream = chat(
            model='deepseek-r1:32b',
            messages=st.session_state.messages,
            #https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            options={'temperature': 0.65, 'top_p': 0.8, 'top_k': 50,'num_ctx': 15000},
            stream=True
        )
        
        # Process streaming chunks
        for chunk in stream:
            if chunk.message.content:
                full_response += chunk.message.content
                response_placeholder.markdown(full_response + "▌")
        
        # Final update without cursor
        response_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({'role': 'assistant', 'content': full_response})