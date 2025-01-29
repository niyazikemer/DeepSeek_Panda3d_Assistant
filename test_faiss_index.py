from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the FAISS index
embeddings = HuggingFaceEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Test the index with a query
query = "how to import blender files into Panda3d"  # Replace with a query relevant to your documents
relevant_docs = faiss_index.similarity_search(query, k=2)  # Retrieve top 2 documents

# Print the results
for doc in relevant_docs:
    print(doc.page_content)
    