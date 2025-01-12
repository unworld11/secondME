from embed import Config, DocumentIndexer, Path

from dataclasses import dataclass
from typing import List, Dict, Any,Optional

@dataclass
class SearchResult:
    source: str
    content: str
    metadata: Dict[str, Any]
    score: float

class QueryEngine:
    def __init__(self, indexer):
        self.indexer = indexer

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        results = self.indexer.search(query, num_results=num_results)
        search_results = []
        
        if results["documents"]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                search_results.append(
                    SearchResult(
                        source=metadata["source"],
                        content=doc[:200],
                        metadata=metadata,
                        score=1.0 - distance  # Convert distance to similarity score
                    )
                )
        
        return search_results

class IndexerFactory:
    _instance = None

    @classmethod
    def get_indexer(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            from embed import DocumentIndexer, Config
            config = Config(persist_dir=config_path or "chroma_db")
            cls._instance = DocumentIndexer(config=config)
        return cls._instance

def query_documents(query: str, config_path: Optional[str] = None) -> List[SearchResult]:
    indexer = IndexerFactory.get_indexer(config_path)
    engine = QueryEngine(indexer)
    return engine.search(query)