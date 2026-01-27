"""
Example usage of Mistral RAG integration
"""

from mistral_rag import MistralRAG
from enhanced_document_processor import EnhancedDocumentProcessor
from langchain_vector_store import LangChainVectorStore

def setup_rag_system(mistral_api_key: str):
    """Set up the RAG system with Mistral API"""
    # Initialize components
    doc_processor = EnhancedDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    vector_store = LangChainVectorStore(
        embedding_model="all-MiniLM-L6-v2"
    )
    
    mistral_rag = MistralRAG(api_key=mistral_api_key)
    
    # Test API connection
    if not mistral_rag.test_connection():
        raise Exception("Failed to connect to Mistral API")
    
    return doc_processor, vector_store, mistral_rag

def process_query(query: str, vector_store, mistral_rag, k: int = 5):
    """Process a query using the RAG system"""
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=k)
    
    # Format documents for Mistral
    context_docs = []
    for doc in docs:
        context_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "metadata": doc.metadata
        })
    
    # Generate response using Mistral
    response = mistral_rag.generate_response(query, context_docs)
    return response

def main():
    # Your Mistral API key
    MISTRAL_API_KEY = "ag:b4b132cf:20251002:untitled-agent:2f1eefae"
    
    try:
        # Set up the system
        print("🚀 Setting up RAG system...")
        doc_processor, vector_store, mistral_rag = setup_rag_system(MISTRAL_API_KEY)
        
        # Example usage
        while True:
            query = input("\nEnter your question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            print("\n🔍 Processing query...")
            response = process_query(query, vector_store, mistral_rag)
            
            if "error" in response:
                print(f"\n❌ Error: {response['error']}")
            else:
                print("\n📝 Answer:")
                print(response["answer"])
                print("\n📚 Sources:")
                for source in response["sources"]:
                    print(f"- {source}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()