import ollama

# List available models
models = ollama.list()
print("Available models:", models)

# Generate a response using the DeepSeek model
response = ollama.generate(model="deepseek-r1:32b", prompt="What is the capital of France?")
print("Response:", response["response"])