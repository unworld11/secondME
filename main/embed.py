import os
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
from datetime import datetime
import pandas as pd
import chromadb
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings

class Config:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.collection_name = "documents"
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.persist_dir = persist_dir
        self.data_dir = "database"
        self.log_dir = "logs"
        self.supported_extensions = {'.html', '.csv'}

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
        self.config = config
        Path(config.persist_dir).mkdir(parents=True, exist_ok=True)
        self.tracker = FileTracker(config.log_dir)
        self.client = chromadb.PersistentClient(path=config.persist_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.model_name)
        self.collection = self._get_collection(config.collection_name)

    def _extract_csv_text(self, file_path: Path) -> str:
        df = pd.read_csv(file_path)
        return ' '.join(df.astype(str).values.flatten())

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
        if file_path.suffix == '.html':
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        elif file_path.suffix == '.csv':
            return self._extract_csv_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    def _find_files(self, directory: Path) -> Set[Path]:
        files = set()
        for ext in self.config.supported_extensions:
            files.update(directory.rglob(f"*{ext}"))
        return files
    
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
        files = self._find_files(directory)
        logging.info(f"Found {len(files)} files to process")
        
        for file_path in sorted(files):
            try:
                doc_id = f"doc_{file_path.name}"
                if doc_id in indexed_docs:
                    continue

                text = self._extract_text(file_path)
                metadata = {
                    "source": str(file_path),
                    "file_type": file_path.suffix[1:],
                    "filename": file_path.name
                }
                
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata],
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

