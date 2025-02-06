from langchain_community.document_loaders import UnstructuredRSTLoader, TextLoader
from langchain_core.documents import Document
import os
from typing import List

class CustomDocumentLoader:
    def __init__(self, source_dir: str):
        self.source_dir = source_dir
        
    def load_rst_files(self) -> List[Document]:
        documents = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.rst'):
                    file_path = os.path.join(root, file)
                    try:
                        # Use 'single' mode to load entire file
                        loader = UnstructuredRSTLoader(file_path=file_path, mode="single")
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading RST file {file_path}: {e}")
        return documents

    def load_python_files(self) -> List[Document]:
        documents = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading Python file {file_path}: {e}")
        return documents
        
    def load_all(self) -> List[Document]:
        documents = []
        documents.extend(self.load_python_files())
        documents.extend(self.load_rst_files())
        return documents

# Usage example
if __name__ == "__main__":
    loader = CustomDocumentLoader("_test_source_code")
    documents = loader.load_all()
    print(f"Loaded {len(documents)} documents")
    print(documents[0])