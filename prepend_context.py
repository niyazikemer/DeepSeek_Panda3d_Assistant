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
import hashlib
import google.generativeai as genai


from langchain_core.embeddings import Embeddings


class DocumentIndexer:
    def __init__(self, output_dir: str = "processed_documents/prepended_chunks", 
                 index_path: str = "faiss_index", chunk_size: int = 1000, 
                 chunk_overlap: int = 200):        
        self.output_dir = output_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.documents = []
        # self.chunks = []
        self.vectorstore = None      
 

    def generate_context(self, document_content: str, chunk_content: str) -> str:
        prompt = f"""
        Here is the whole document
        <document>
        {document_content}
        </document>
        and here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Use doc strings to understand. Answer only with the succinct context and nothing else.
        """
        response = ollama.generate(model="deepseek-r1:32b", prompt=prompt)
        return response['response'].split('</think>')[-1].strip()




    def load_json_docs(self, json_doc_dir: str):
        """Load previously processed chunks and convert to Document objects"""
        document_type = json_doc_dir.split('/')[-1]
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
        print(f"Loaded {len(json_doc)} {document_type}")
        return json_doc
    
    def process_and_save_chunks(self, chunks: List[Document], context_docs: List[Document]) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        doc_map = {doc.metadata['doc_id']: doc for doc in context_docs}
        
        for chunk in chunks:
            # Process chunk
            doc_id = chunk.metadata['doc_id']
            original_doc = doc_map[doc_id]
            context = self.generate_context(original_doc.page_content, chunk.page_content)
            chunk.page_content = f"{context}\n\n{chunk.page_content}"
            
            # Use existing chunk number from metadata
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            chunk_num = chunk.metadata['chunk_number']
            file_name = f"chunk_{chunk_num:05d}_{content_hash}.json"
            save_path = os.path.join(self.output_dir, file_name)
            
            with open(save_path, 'w') as f:
                chunk_dict = {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                }
                json.dump(chunk_dict, f, indent=2)
                
        return self.output_dir

      
    def create_index(self, json_doc_dir: str = None):
        """Create index from previously processed chunks"""
        if json_doc_dir is None:
            json_doc_dir = self.output_dir
            
        print("Loading chunks...")
        chunks = self.load_json_docs(json_doc_dir)
        print(f"Loaded {len(chunks)} chunks")

        print("Creating embeddings...")
        embeddings = GeminiEmbeddings()
    #     embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2",  # Add model name
    #     model_kwargs={'device': 'cuda'},
    #     encode_kwargs={'batch_size': 8}
    # )
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

class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "models/text-embedding-004"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        result = genai.embed_content(
            model=self.model,
            content=texts
        )
        #print(result['embedding'])
        return result['embedding']

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query"""
        result = genai.embed_content(
            model=self.model,
            content=[text]
        )
        #print(result['embedding'][0])
        return result['embedding'][0]

indexer = DocumentIndexer(    
    output_dir="processed_documents/prepended_chunks",
    index_path="faiss_index",
    chunk_size=1000,
    chunk_overlap=200
)

if __name__ == "__main__":
    # raw_chunks = indexer.load_json_docs('processed_documents/chunks/set_0')
    indexer.create_index()
    # context_documents = indexer.load_json_docs('processed_documents/context_documents')
    # # #print(raw_chunks[0])
    # indexer.process_and_save_chunks(raw_chunks, context_documents)
    #chunks_dir = indexer.process_and_save_chunks(raw_chunks, context_documents)

    # raw_chunks = indexer.load_json_docs('processed_documents/chunks/set_1')
    # indexer.process_and_save_chunks(raw_chunks, context_documents)
    # raw_chunks = indexer.load_json_docs('processed_documents/chunks/set_2')
    # indexer.process_and_save_chunks(raw_chunks, context_documents)
    # raw_chunks = indexer.load_json_docs('processed_documents/chunks/set_3')
    # indexer.process_and_save_chunks(raw_chunks, context_documents)