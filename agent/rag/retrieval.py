import os
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class DocumentChunk:
    def __init__(self, id: str, content: str, source: str, metadata: Optional[Dict] = None):
        self.id = id
        self.content = content
        self.source = source
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"DocumentChunk(id={self.id}, source={self.source}, length={len(self.content)})"


class SimpleRetriever:
    
    def __init__(self, docs_dir: str, chunk_size: int = 500):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self._load_documents()
    
    def _chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunk_idx = 0
        for para in paragraphs:
            if len(para) < 20:
                continue
            if len(para) > self.chunk_size:
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) < self.chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunk_id = f"{source.replace('.md', '')}::chunk{chunk_idx}"
                            chunks.append(DocumentChunk(chunk_id, current_chunk.strip(), source))
                            chunk_idx += 1
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunk_id = f"{source.replace('.md', '')}::chunk{chunk_idx}"
                    chunks.append(DocumentChunk(chunk_id, current_chunk.strip(), source))
                    chunk_idx += 1
            else:
                chunk_id = f"{source.replace('.md', '')}::chunk{chunk_idx}"
                chunks.append(DocumentChunk(chunk_id, para, source))
                chunk_idx += 1
        
        return chunks
    
    def _load_documents(self):
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")
        
        md_files = [f for f in os.listdir(self.docs_dir) if f.endswith('.md')]
        
        if not md_files:
            raise ValueError(f"No .md files found in {self.docs_dir}")
        
        for filename in md_files:
            filepath = os.path.join(self.docs_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_chunks = self._chunk_text(content, filename)
                self.chunks.extend(doc_chunks)
                
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        
        print(f"Loaded {len(self.chunks)} chunks from {len(md_files)} documents")
        
        if self.chunks:
            texts = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict]:
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            if score < min_score:
                continue
            
            chunk = self.chunks[idx]
            results.append({
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "score": score
            })
        
        return results
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 3) -> List[Dict]:
        matching_chunks = []
        
        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            
            if matches > 0:
                matching_chunks.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "score": matches / len(keywords)  
                })
        
        # Sort by score
        matching_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return matching_chunks[:top_k]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        return self.chunks
    
    def stats(self) -> Dict:
        sources = {}
        for chunk in self.chunks:
            if chunk.source not in sources:
                sources[chunk.source] = 0
            sources[chunk.source] += 1
        
        return {
            "total_chunks": len(self.chunks),
            "total_docs": len(sources),
            "chunks_per_doc": sources,
            "avg_chunk_length": sum(len(c.content) for c in self.chunks) / len(self.chunks) if self.chunks else 0
        }