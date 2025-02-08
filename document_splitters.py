from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from typing import List
import os
import json
import datetime
import hashlib
class CustomDocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        python_docs = []
        rst_docs = []
        
        # Separate documents by type
        for doc in documents:
            if doc.metadata.get('doc_type') == 'python':
                python_docs.append(doc)
            elif doc.metadata.get('doc_type') == 'rst':
                rst_docs.append(doc)
        
        # Initialize language-specific splitters
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        rst_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RST,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Split and maintain metadata
        chunks = []
        
        
    # Process Python documents
        for doc in python_docs:
            doc_chunks = python_splitter.split_documents([doc])
            for chunk in doc_chunks:
                chunk.metadata = {
                    'doc_type': doc.metadata['doc_type'],
                    'doc_id': doc.metadata['doc_id'],
                    'parent': doc.metadata['file_path']
                }
            chunks.extend(doc_chunks)

        # Process RST documents
        for doc in rst_docs:
            doc_chunks = rst_splitter.split_documents([doc])
            for chunk in doc_chunks:
                chunk.metadata = {
                    'doc_type': doc.metadata['doc_type'],
                    'doc_id': doc.metadata['doc_id'],
                    'parent': doc.metadata['file_path']
                }
            chunks.extend(doc_chunks)

        return chunks

    def save_chunks(self, chunks: List[Document], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_number'] = i  # Add global chunk number
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            file_name = f"chunk_{i}_{content_hash}.json"
            save_path = os.path.join(output_dir, file_name)
            
            chunk_dict = {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            }
            with open(save_path, 'w') as f:
                json.dump(chunk_dict, f, indent=2)
        
        return output_dir
        





# Usage example
if __name__ == "__main__":
    from document_loaders import CustomDocumentLoader
    
    loader = CustomDocumentLoader("corpus_panda3d")
    documents = loader.load_save_all()
    
    splitter = CustomDocumentSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    splitter.save_chunks(chunks,'processed_documents/chunks')
    
    print(f"Original documents: {len(documents)}")
    print(f"After splitting: {len(chunks)}")
    # for nm,i in enumerate(chunks):
    #     print(f'Document no:{nm},\n content{i}')