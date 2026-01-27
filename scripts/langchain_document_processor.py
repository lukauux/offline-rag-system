"""
Enhanced Document Processing with LangChain
Uses LangChain's document loaders and text splitters for better chunking.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    print("⚠️  LangChain not installed. Run: pip install langchain langchain-community")

class LangChainDocumentProcessor:
    """Process documents using LangChain loaders and splitters"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        
        # Initialize text splitter with smart chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        print(f"✅ Document Processor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF using LangChain loader"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Enhance metadata
            filename = Path(file_path).name
            for doc in split_docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "pdf",
                    "file_path": file_path
                })
            
            print(f"✓ Processed PDF: {filename} ({len(split_docs)} chunks)")
            return split_docs
            
        except Exception as e:
            print(f"✗ Error processing PDF {file_path}: {str(e)}")
            return []
    
    def process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX using LangChain loader"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Enhance metadata
            filename = Path(file_path).name
            for doc in split_docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "docx",
                    "file_path": file_path
                })
            
            print(f"✓ Processed DOCX: {filename} ({len(split_docs)} chunks)")
            return split_docs
            
        except Exception as e:
            print(f"✗ Error processing DOCX {file_path}: {str(e)}")
            return []
    
    def process_text(self, file_path: str) -> List[Document]:
        """Process text file using LangChain loader"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Enhance metadata
            filename = Path(file_path).name
            for doc in split_docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "text",
                    "file_path": file_path
                })
            
            print(f"✓ Processed TXT: {filename} ({len(split_docs)} chunks)")
            return split_docs
            
        except Exception as e:
            print(f"✗ Error processing TXT {file_path}: {str(e)}")
            return []
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process any supported document file"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.process_docx(file_path)
        elif ext == '.txt':
            return self.process_text(file_path)
        else:
            print(f"⚠️  Unsupported file type: {ext}")
            return []
    
    def create_document_from_text(self, 
                                 text: str, 
                                 metadata: Dict[str, Any]) -> List[Document]:
        """Create and split documents from raw text"""
        doc = Document(page_content=text, metadata=metadata)
        return self.text_splitter.split_documents([doc])

# Test function
if __name__ == "__main__":
    processor = LangChainDocumentProcessor()
    print("\n" + "="*50)
    print("LangChain Document Processor ready!")
    print("="*50)
