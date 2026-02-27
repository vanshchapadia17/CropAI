from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths relative to this file's location (src/RAG/)
RAG_DIR = Path(__file__).parent
RICE_DIR = RAG_DIR / "rice_dataset"
SUGARCANE_DIR = RAG_DIR / "sugarcane_dataset"


def load_txt_files(folder: Path, crop_type: str) -> List[Document]:
    """Load all .txt files from a folder and tag each with crop_type metadata."""
    documents = []
    for txt_file in sorted(folder.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source_file": txt_file.name,
                        "crop_type": crop_type,
                    },
                ))
        except Exception as e:
            print(f"Warning: could not read {txt_file.name}: {e}")
    return documents


def load_all_documents() -> List[Document]:
    """Load all scraped txt files for rice and sugarcane."""
    rice_docs = load_txt_files(RICE_DIR, "rice")
    sugarcane_docs = load_txt_files(SUGARCANE_DIR, "sugarcane")
    all_docs = rice_docs + sugarcane_docs
    print(
        f"Loaded {len(rice_docs)} rice docs + {len(sugarcane_docs)} sugarcane docs "
        f"= {len(all_docs)} total"
    )
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks
