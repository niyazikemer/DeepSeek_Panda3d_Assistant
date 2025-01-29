# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain.schema import Document
import os
# Load your documents
#documents = ["doc1 the creator of this system is Niyazi", "doc2 Isik is the daughter of Niyazi", "doc3 text"]
# Load documents from collected_docs directory
documents = []
for root, _, files in os.walk('converted_txt'):
    for file in files:
        if file.endswith('.txt'):
            txt_path = os.path.join(root, file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())


# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create FAISS index
faiss_index = FAISS.from_texts(documents, embeddings)

# Save the index
faiss_index.save_local("faiss_index")