from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from typing import List
import os
import json
import datetime
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
        
        # Split documents and maintain metadata
        chunks = []
        chunks.extend(python_splitter.split_documents(python_docs))
        chunks.extend(rst_splitter.split_documents(rst_docs))
        
        return chunks
    
    def save_chunks(self, chunks: List[Document], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, f"{timestamp}_raw_chunks")
        os.makedirs(save_dir, exist_ok=True)

        for i, chunk in enumerate(chunks):
            file_path = os.path.join(save_dir, f"raw_chunk_{i}.json")
            with open(file_path, 'w') as f:
                chunk_dict = {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                }
                json.dump(chunk_dict, f, indent=2)
        
        return save_dir
        





# Usage example
if __name__ == "__main__":
    from document_loaders import CustomDocumentLoader
    
    loader = CustomDocumentLoader("_test_source_code")
    documents = loader.load_all()
    
    splitter = CustomDocumentSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    splitter.save_chunks(chunks,'processed_documents')
    
    print(f"Original documents: {len(documents)}")
    print(f"After splitting: {len(chunks)}")
    # for nm,i in enumerate(chunks):
    #     print(f'Document no:{nm},\n content{i}')