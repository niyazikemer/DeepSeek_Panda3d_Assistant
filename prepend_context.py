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
    def __init__(self, source_dir: str, output_dir: str = "processed_documents", 
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
        
    @staticmethod
    def get_supported_extensions():
        return [
            ".py", ".js", ".java", ".cpp", ".cs", ".go", ".kt", 
            ".lua", ".pl", ".rb", ".rs", ".scala", ".ts"
        ]

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
            context = self.generate_context(original_doc.page_content, chunk.page_content)
            new_content = f"{context}\n\n{chunk.page_content}"
            chunk.page_content = new_content
            processed_chunks.append(chunk)
        return processed_chunks

    def save_documents(self, documents, subdir_prefix=""):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, f"{timestamp}_{subdir_prefix}")
        os.makedirs(save_dir, exist_ok=True)

        docs_by_source = {}
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'unknown_source')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'chunk_index': i
            })

        for source, docs in docs_by_source.items():
            base_name = os.path.basename(source)
            file_path = os.path.join(save_dir, f"{base_name}_processed.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
        return save_dir

    def load_source_documents(self):
        loader = GenericLoader.from_filesystem(
            self.source_dir,
            glob="**/*",
            suffixes=self.get_supported_extensions(),
            parser=LanguageParser(parser_threshold=5)
        )
        
        text_loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.txt",
            recursive=True
        )
        
        try:
            self.documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Error loading source code files: {e}")
        
        try:
            self.documents.extend(text_loader.load())
        except Exception as e:
            print(f"Warning: Error loading text files: {e}")

    def split_documents(self):
        language_docs = {}
        other_docs = []
        
        for doc in self.documents:
            if 'language' in doc.metadata:
                lang = doc.metadata['language']
                if lang not in language_docs:
                    language_docs[lang] = []
                language_docs[lang].append(doc)
            else:
                other_docs.append(doc)
        
        for lang, docs in language_docs.items():
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.chunks.extend(splitter.split_documents(docs))
        
        if other_docs:
            default_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.chunks.extend(default_splitter.split_documents(other_docs))

        self.chunks = self.process_chunks_with_context()
        chunks_dir = self.save_documents(self.chunks, "chunks")
        print(f"Saved {len(self.chunks)} chunks to {chunks_dir}")

    def process_and_save_chunks(self):
        """Process documents into chunks and save them"""
        print("Loading documents...")
        self.load_source_documents()
        print(f"Loaded {len(self.documents)} documents")
        
        orig_dir = self.save_documents(self.documents, "original")
        print(f"Saved original documents to {orig_dir}")
        
        print("Splitting documents...")
        self.split_documents()
        return self.chunks
    
    def load_chunks(self, chunks_dir: str):
        """Load previously processed chunks and convert to Document objects"""
        chunks = []
        for root, _, files in os.walk(chunks_dir):
            for file in files:
                if file.endswith('_processed.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        file_chunks = json.load(f)
                        for chunk in file_chunks:
                            # Convert dict to Document object
                            doc = Document(
                                page_content=chunk['content'],
                                metadata=chunk['metadata']
                            )
                            chunks.append(doc)
        return chunks  # Add return statement
       
    def create_index(self, chunks_dir: str):
        """Create index from previously processed chunks"""
        print("Loading chunks...")
        chunks = self.load_chunks(chunks_dir)
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
# chunks = indexer.process_and_save_chunks()

chunks_dir = "processed_documents/20250204_065433_chunks"  # Use actual directory
indexer.create_index(chunks_dir)