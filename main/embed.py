import os
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
from datetime import datetime

import chromadb
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings

class Config:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.collection_name = "html_documents"
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.persist_dir = persist_dir
        self.log_dir = "logs"

class FileTracker:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        log_file = self.log_dir / f"indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

class DocumentIndexer:
    def __init__(self, config: Config):
        Path(config.persist_dir).mkdir(parents=True, exist_ok=True)
        self.tracker = FileTracker(config.log_dir)
        
        self.client = chromadb.PersistentClient(path=config.persist_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.model_name)
        self.collection = self._get_collection(config.collection_name)

    def _get_collection(self, name: str):
        try:
            return self.client.get_collection(
                name=name,
                embedding_function=self.embeddings.embed_documents
            )
        except ValueError:
            return self.client.create_collection(
                name=name,
                embedding_function=self.embeddings.embed_documents
            )

    def _extract_text(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    def _find_html_files(self, directory: Path) -> Set[Path]:
        html_files = set()
        
        # Method 1: Using rglob
        html_files.update(directory.rglob("*.html"))
        
        # Method 2: Using os.walk as backup
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.html'):
                    file_path = Path(root) / file
                    html_files.add(file_path)
        
        return html_files

    def index_directory(self, directory: Path) -> None:
        start_time = datetime.now()
        directory = Path(directory)
        logging.info(f"Starting indexing of directory: {directory}")
        
        indexed_docs = self.get_indexed_docs()
        html_files = self._find_html_files(directory)
        
        logging.info(f"Found {len(html_files)} HTML files")
        
        for file_path in sorted(html_files):
            try:
                doc_id = f"doc_{file_path.name}"
                if doc_id in indexed_docs:
                    logging.debug(f"Skipping already indexed file: {file_path}")
                    continue

                text = self._extract_text(file_path)
                self.collection.add(
                    documents=[text],
                    metadatas=[{"source": str(file_path)}],
                    ids=[doc_id]
                )
                logging.info(f"Successfully indexed: {file_path}")
                
            except (IOError, OSError) as e:
                logging.error(f"Failed to read {file_path}: {e}")
            except chromadb.errors.ChromaDBError as e:
                logging.error(f"Database error for {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error processing {file_path}: {e}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.info(f"Indexing completed in {duration:.2f} seconds")
        logging.info(f"Total files processed: {len(html_files)}")

    def get_indexed_docs(self) -> Set[str]:
        try:
            return {doc_id for doc_id in self.collection.get()["ids"]}
        except chromadb.errors.ChromaDBError:
            logging.error("Failed to retrieve indexed documents")
            return set()

    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=num_results
            )
        except chromadb.errors.ChromaDBError as e:
            logging.error(f"Search failed: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

