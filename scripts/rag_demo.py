"""
Complete RAG system demo with document processing and Mistral API
"""

import os
from pathlib import Path
from mistral_rag import MistralRAG
from simple_document_processor import SimpleDocumentProcessor
from langchain_vector_store import LangChainVectorStore

class RAGDemo:
    def __init__(self, mistral_api_key: str):
        """Initialize the RAG system"""
        print("\n🚀 Initializing RAG system...")
        
        # Initialize components
        self.doc_processor = SimpleDocumentProcessor(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.vector_store = LangChainVectorStore(
            embedding_model="all-MiniLM-L6-v2"
        )
        
        self.mistral = MistralRAG(api_key=mistral_api_key)
        
        # Create data directory if it doesn't exist
        self.data_dir = Path(".data")
        self.data_dir.mkdir(exist_ok=True)
        
        print("✅ RAG system initialized")
    
    def process_document(self, file_path: str) -> bool:
        """Process a document and add it to the vector store"""
        try:
            print(f"\n📑 Processing document: {Path(file_path).name}")
            
            # Process document
            docs = self.doc_processor.process_file(file_path)
            
            if not docs:
                print("❌ No content extracted from document")
                return False
            
            # Add to vector store
            self.vector_store.add_documents(docs)
            
            print(f"✅ Successfully processed document ({len(docs)} chunks)")
            return True
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return False
    
    def query(self, question: str, k: int = 3):
        """Query the RAG system"""
        try:
            print(f"\n❓ Question: {question}")
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=k)
            
            if not docs:
                print("❌ No relevant information found")
                return
            
            # Format documents for Mistral
            context_docs = []
            for doc in docs:
                context_docs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None)
                })
            
            # Generate response
            print("\n🤔 Generating response...")
            response = self.mistral.generate_response(question, context_docs)
            
            # Print response
            print("\n🤖 Answer:")
            print(response["answer"])
            print("\n📚 Sources:")
            for source in response["sources"]:
                page_info = f" (Page {docs[0].metadata['page']})" if docs[0].metadata.get('page') else ""
                print(f"- {source}{page_info}")
                
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")

def main():
    # Mistral API key
    MISTRAL_API_KEY = "WG5uIO1P7Qq2SHGgspZ7asmq0M5HEUKi"
    
    # Initialize RAG system
    rag = RAGDemo(MISTRAL_API_KEY)
    
    while True:
        print("\n" + "="*50)
        print("RAG System Demo")
        print("1. Process a document")
        print("2. Ask a question")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            file_path = input("\nEnter the path to the document: ")
            if os.path.exists(file_path):
                rag.process_document(file_path)
            else:
                print("❌ File not found")
                
        elif choice == "2":
            question = input("\nEnter your question: ")
            rag.query(question)
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice")

if __name__ == "__main__":
    main()