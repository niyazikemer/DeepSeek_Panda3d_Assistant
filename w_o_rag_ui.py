import streamlit as st
import ollama

# Initialize chat history and system prompt
if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = """You are a helpful AI assistant. Always provide accurate, 
    informative, and engaging responses while maintaining a conversational tone."""
    st.session_state.messages.append({"role": "system", "content": system_prompt})

# Set page title
st.title("Chatbot with DeepSeek")

def get_ai_response(query):
    # Format entire conversation history
    formatted_prompt = ""
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            formatted_prompt += f"Instructions: {msg['content']}\n\n"
        else:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted_prompt += f"{role}: {msg['content']}\n"
    
    formatted_prompt += f"Human: {query}\nAssistant:"
    
    response = ollama.generate(
        model="deepseek-r1:32b",
        prompt=formatted_prompt,
        options={
            "temperature": 0.7,
            "top_p": 0.95
        }
    )
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