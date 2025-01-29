from langchain.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def load_documents():
    # Initialize loaders for PDFs and other supported formats
    loader = DirectoryLoader(
        'converted_txt',
        glob='**/*.txt',  # Adjust based on file types
        loader_cls=TextLoader,
    )

    # Load documents from the directory
    documents = loader.load()

    return documents

# Split text into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Size of each document chunk
    chunk_overlap=200,  # Overlap between consecutive chunks
)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings()

# Process documents into document chunks
document_chunks = text_splitter.split_documents(
    load_documents()
)

# Create FAISS vector store from document chunks and embeddings
vectorstore = FAISS.from_documents(
    document_chunks,
    embedding=embeddings,
)
vectorstore.save_local("faiss_index")





