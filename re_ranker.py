# Add to your imports
from FlagEmbedding import FlagReranker
import torch  # For FP16 optimization
from langchain_core.documents import Document
from typing import List, Tuple

class OptimizedReranker:
    def __init__(self, model_name='BAAI/bge-reranker-large'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = FlagReranker(
            model_name,
            use_fp16=True,  # FP16 for 2x speed + half memory
            device=self.device
        )
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 20) -> List[Document]:
        """Rerank documents with batch processing"""
        # Batch processing for efficiency
        batch_size = 32  # Adjust based on your GPU memory
        pairs = [(query, doc.page_content[:512]) for doc in documents]  # Truncate to 512 tokens
        
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores.extend(self.reranker.compute_score(batch))
        
        # Combine scores with documents
        scored_docs = list(zip(scores, documents))
        
        # Sort and select top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

# Initialize during app setup
#reranker = OptimizedReranker()