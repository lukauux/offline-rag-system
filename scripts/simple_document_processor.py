"""
Simple document processor for RAG system
"""

from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SimpleDocumentProcessor:
    """Process documents for RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print(f"✅ Document Processor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file and return chunks"""
        try:
            file_path = Path(file_path)
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            texts = self.text_splitter.split_text(content)
            
            # Create documents
            docs = []
            for i, text in enumerate(texts):
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': file_path.name,
                        'chunk': i,
                        'file_type': file_path.suffix.lower()[1:],
                    }
                )
                docs.append(doc)
            
            print(f"✅ Processed {file_path.name} into {len(docs)} chunks")
            return docs
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return []