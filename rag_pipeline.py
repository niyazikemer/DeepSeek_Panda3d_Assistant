from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import ollama

# Load the FAISS index
embeddings = HuggingFaceEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def rag_pipeline(query):
    """
    RAG Pipeline:
    1. Retrieve relevant documents using FAISS.
    2. Combine the documents with the query to create a prompt.
    3. Generate a response using the DeepSeek model via Ollama.
    """
    # Step 1: Retrieve relevant documents
    relevant_docs = faiss_index.similarity_search(query, k=5)  # Retrieve top 5 documents
    
    # Step 2: Combine documents with the query
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Step 3: Generate response using DeepSeek via Ollama
    response = ollama.generate(model="deepseek-r1:32b", prompt=prompt)
    return response["response"]

# Example usage
if __name__ == "__main__":
    user_query = "Who is Niyazi"  # Replace with your query
    answer = rag_pipeline(user_query)
    print("Answer:", answer)