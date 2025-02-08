from langchain_community.document_loaders import UnstructuredRSTLoader, TextLoader
from langchain_core.documents import Document
import os
from typing import List
import datetime
import json
import hashlib

class CustomDocumentLoader:
    def __init__(self, source_dir: str, output_dir: str = "processed_documents/context_documents"):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.doc_counter = 0
        
    def _get_next_doc_id(self) -> str:
        doc_id = f"doc_{self.doc_counter}"
        self.doc_counter += 1
        return doc_id
    
    def save_documents(self, documents: List[Document]):
        os.makedirs(self.output_dir, exist_ok=True)

        for doc in documents:
            source_file = os.path.basename(doc.metadata['file_path'])
            base_name = os.path.splitext(source_file)[0]
            
            # Create hash from content
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            new_file_name = f"{content_hash}_{base_name}.json"
            save_path = os.path.join(self.output_dir, new_file_name)
            
            # Update metadata with new file path
            doc.metadata['file_path'] = save_path
            
            with open(save_path, 'w') as f:
                doc_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                json.dump(doc_dict, f, indent=2)
        
        return self.output_dir
    
    def load_python_files(self) -> List[Document]:
        documents = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc_id = self._get_next_doc_id()
                            doc.metadata.update({
                                'doc_type': 'python',                                
                                'file_path': file_path,
                                'doc_id': doc_id
                            })
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading Python file {file_path}: {e}")
        return documents

    def load_rst_files(self) -> List[Document]:
        documents = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.rst'):
                    file_path = os.path.join(root, file)
                    try:
                        loader = UnstructuredRSTLoader(file_path=file_path, mode="single")
                        docs = loader.load()
                        for doc in docs:
                            doc_id = self._get_next_doc_id()
                            doc.metadata.update({
                                'doc_type': 'rst',                                
                                'file_path': file_path,
                                'doc_id': doc_id
                            })
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading RST file {file_path}: {e}")
        return documents
        
    def load_save_all(self) -> List[Document]:
        documents = []
        documents.extend(self.load_python_files())
        documents.extend(self.load_rst_files())
        save_dir = self.save_documents(documents)
        print(f"Saved raw documents to {save_dir}")
        return documents

# Usage example
if __name__ == "__main__":
    loader = CustomDocumentLoader("corpus_panda3d")
    documents = loader.load_save_all()
    print(f"Loaded {len(documents)} documents")
    #print(documents[1])