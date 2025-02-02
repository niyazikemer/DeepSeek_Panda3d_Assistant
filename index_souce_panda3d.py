from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
import datetime

def get_supported_extensions():
    """Returns a list of supported file extensions for code parsing."""
    return [
        ".py", ".js", ".java", ".cpp", ".cs", ".go", ".kt", 
        ".lua", ".pl", ".rb", ".rs", ".scala", ".ts"
    ]

def save_documents(documents, output_dir="processed_documents", subdir_prefix=""):
    """
    Save processed documents to files for inspection.
    
    Args:
        documents: List of Document objects
        output_dir: Directory to save the documents
        subdir_prefix: Prefix for the subdirectory name (e.g., "original" or "chunks")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subdirectory
    save_dir = os.path.join(output_dir, f"{timestamp}_{subdir_prefix}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Group documents by source file
    docs_by_source = {}
    for i, doc in enumerate(documents):
        source = doc.metadata.get('source', 'unknown_source')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'chunk_index': i  # Add index to track chunk order
        })
    
    # Save each source file's documents
    for source, docs in docs_by_source.items():
        base_name = os.path.basename(source)
        file_name = f"{base_name}_processed.json"
        file_path = os.path.join(save_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
    
    return save_dir

def load_source_documents(source_dir):
    """
    Load source code files using language-aware parsing.
    
    Args:
        source_dir: Directory containing source code files
    """
    # Load source code files with language parsing
    loader = GenericLoader.from_filesystem(
        source_dir,
        glob="**/*",  # Recursive search in subdirectories
        suffixes=get_supported_extensions(),
        parser=LanguageParser(parser_threshold=50)  # Only parse files > 50 lines
    )
    
    # For any text files that aren't source code
    text_loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        recursive=True
    )
    
    documents = []
    try:
        documents.extend(loader.load())
    except Exception as e:
        print(f"Warning: Error loading source code files: {e}")
    
    try:
        documents.extend(text_loader.load())
    except Exception as e:
        print(f"Warning: Error loading text files: {e}")
        
    return documents

def split_documents(documents, default_chunk_size=1000, default_chunk_overlap=200):
    """
    Split documents using language-specific splitters when possible.
    """
    # Group documents by language
    language_docs = {}
    other_docs = []
    
    for doc in documents:
        if 'language' in doc.metadata:
            lang = doc.metadata['language']
            if lang not in language_docs:
                language_docs[lang] = []
            language_docs[lang].append(doc)
        else:
            other_docs.append(doc)
    
    # Split documents using language-specific splitters
    chunks = []
    for lang, docs in language_docs.items():
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap
        )
        chunks.extend(splitter.split_documents(docs))
    
    # Split remaining documents with default splitter
    if other_docs:
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap
        )
        chunks.extend(default_splitter.split_documents(other_docs))
    
    # Save the chunks with a specific prefix
    chunks_dir = save_documents(chunks, "processed_documents", "chunks")
    print(f"Saved {len(chunks)} chunks to {chunks_dir}")
    
    return chunks

def create_index(source_dir, index_path="faiss_index"):
    """
    Create a searchable index from source code files.
    
    Args:
        source_dir: Directory containing source code files
        index_path: Path to save the FAISS index
    """
    print("Loading documents...")
    documents = load_source_documents(source_dir)
    print(f"Loaded {len(documents)} documents")
    
    # Save original documents
    orig_dir = save_documents(documents, "processed_documents", "original")
    print(f"Saved original documents to {orig_dir}")
    
    print("Splitting documents...")
    document_chunks = split_documents(documents)
    print(f"Created {len(document_chunks)} chunks")
    
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings()
    
    print("Building vector store...")
    vectorstore = FAISS.from_documents(
        document_chunks,
        embedding=embeddings,
    )
    
    print(f"Saving index to {index_path}...")
    vectorstore.save_local(index_path)
    print("Done!")
    
    return vectorstore


if __name__ == "__main__":
    # Example usage
    source_directory = "source_code"
    create_index(source_directory)