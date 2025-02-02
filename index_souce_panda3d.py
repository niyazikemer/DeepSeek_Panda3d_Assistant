from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_supported_extensions():
    """Returns a list of supported file extensions for code parsing."""
    return [
        ".py", ".js", ".java", ".cpp", ".cs", ".go", ".kt", 
        ".lua", ".pl", ".rb", ".rs", ".scala", ".ts"
    ]

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

def create_language_splitter(language, chunk_size=1000, chunk_overlap=200):
    """
    Create a language-specific text splitter.
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

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
        splitter = create_language_splitter(
            lang,
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