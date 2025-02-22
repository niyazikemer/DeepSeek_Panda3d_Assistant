# from langchain_community.embeddings import HuggingFaceEmbeddings
from prepend_context import GeminiEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )
embeddings = GeminiEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

class HybridRetriever:
    def __init__(self, faiss_index, alpha=0.5):
        self.faiss_index = faiss_index
        self.alpha = alpha
        # Store documents with their IDs and create mappings
        self.documents = []
        self.doc_mapping = {}  # doc_id -> index
        self.index_to_doc_id = []  # index -> doc_id
        for i, (doc_id, doc) in enumerate(faiss_index.docstore._dict.items()):
            self.documents.append(doc.page_content)
            self.doc_mapping[doc_id] = i
            self.index_to_doc_id.append(doc_id)
        tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        # Get dense results
        dense_docs = self.faiss_index.similarity_search_with_score(query, k=k)
        
        # Create a mapping of document indices to their dense scores
        dense_scores = {}
        for doc, score in dense_docs:
            doc_id = next(k for k, v in self.faiss_index.docstore._dict.items() if v == doc)
            doc_idx = self.doc_mapping[doc_id]
            dense_scores[doc_idx] = score
        
        # Get sparse results
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        dense_scores_list = [dense_scores.get(i, min(dense_scores.values())) for i in range(len(self.documents))]
        dense_scores_norm = self._normalize_scores(dense_scores_list)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores
        final_scores = []
        for i, _ in enumerate(self.documents):
            dense_score = dense_scores_norm[i]
            sparse_score = bm25_scores_norm[i]
            combined_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            final_scores.append((i, combined_score))
        
        # Get top k documents using the correct doc_ids
        top_k_indices = sorted(final_scores, key=lambda x: x[1], reverse=True)[:k]
        # Retrieve documents using index_to_doc_id mapping
        return [self.faiss_index.docstore._dict[self.index_to_doc_id[idx]] for idx, _ in top_k_indices]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score) if max_score > min_score else 0 for s in scores]
    
# hybrid_retriever = HybridRetriever(faiss_index)

# print(hybrid_retriever.hybrid_search("I want to talk about panda3d clustermsgs", k=10))