from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document 
import os
import json
import datetime
import ollama
from typing import Dict, List

class DocumentIndexer:
    def __init__(self, source_dir: str, output_dir: str = "processed_documents/prepended_chunks", 
                 index_path: str = "faiss_index", chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        self.vectorstore = None      
 

    def generate_context(self, document_content: str, chunk_content: str) -> str:
        prompt = f"""
        <document>
        {document_content}
        </document>
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Use doc strings to understand. Answer only with the succinct context and nothing else.
        """
        response = ollama.generate(model="deepseek-r1:32b", prompt=prompt)
        return response['response'].split('</think>')[-1].strip()

    def process_chunks_with_context(self) -> List:
        processed_chunks = []
        for chunk in self.chunks:
            source = chunk.metadata['source']
            original_doc = next(doc for doc in self.documents if doc.metadata['source'] == source)
            print(f'original_doc: {original_doc.metadata["source"]}')
            context = self.generate_context(original_doc.page_content, chunk.page_content)
            new_content = f"{context}\n\n{chunk.page_content}"
            chunk.page_content = new_content
            processed_chunks.append(chunk)
        return processed_chunks


    def load_json_docs(self, json_doc_dir: str):
        """Load previously processed chunks and convert to Document objects"""
        json_doc = []
        for root, _, files in os.walk(json_doc_dir):
            for file in files:
                if file.endswith('.json'):  # Changed from startswith('chunk_')
                    with open(os.path.join(root, file), 'r') as f:
                        chunk_data = json.load(f)  # Load the entire JSON file
                        # Create Document object from JSON data
                        doc = Document(
                            page_content=chunk_data['content'],
                            metadata=chunk_data['metadata']
                        )
                        json_doc.append(doc)
        print(f"Loaded {len(json_doc)} chunks")
        return json_doc
       
    def create_index(self, json_doc_dir: str):
        """Create index from previously processed chunks"""
        print("Loading chunks...")
        chunks = self.load_chunks(json_doc_dir)
        print(f"Loaded {len(chunks)} chunks")

        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Add model name
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'batch_size': 8}
        )

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if i == 0:
                self.vectorstore = FAISS.from_documents(batch, embedding=embeddings)
            else:
                self.vectorstore.add_documents(batch)

        print(f"Saving index to {self.index_path}...")
        self.vectorstore.save_local(self.index_path)
        print("Done!")
        
        return self.vectorstore

indexer = DocumentIndexer(
    source_dir="_test_source_code",
    output_dir="processed_documents",
    index_path="faiss_index",
    chunk_size=1000,
    chunk_overlap=200
)
raw_chunks = indexer.load_json_docs('processed_documents/chunks')
context_documents = indexer.load_json_docs('processed_documents/context_documents')
print(list(raw_chunks))

#chunks = indexer.process_and_save_chunks()

# json_doc_dir = "processed_documents/20250204_065433_chunks"  # Use actual directory
# indexer.create_index(json_doc_dir)