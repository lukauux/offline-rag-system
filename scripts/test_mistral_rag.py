"""
Simple demo of Mistral API RAG system
"""

from mistral_rag import MistralRAG
from langchain_document_processor import LangChainDocumentProcessor
from langchain_vector_store import LangChainVectorStore

def test_rag_system():
    """Test the RAG system with Mistral API"""
    
    # Initialize components
    print("\n🚀 Initializing RAG system...")
    
    # Mistral API key
    MISTRAL_API_KEY = "WG5uIO1P7Qq2SHGgspZ7asmq0M5HEUKi"
    
    try:
        # Initialize Mistral RAG
        mistral = MistralRAG(api_key=MISTRAL_API_KEY)
        
        # Test connection
        if not mistral.test_connection():
            print("❌ Failed to connect to Mistral API")
            return
        
        print("✅ Connected to Mistral API")
        
        # Example query with some context
        context_docs = [
            {
                "content": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair.",
                "source": "paris_landmarks.txt"
            },
            {
                "content": "The tower is 324 metres (1,063 ft) tall, and was the tallest man-made structure in the world for 41 years until the Chrysler Building in New York City was finished in 1930.",
                "source": "tower_facts.txt"
            }
        ]
        
        # Test query
        query = "When was the Eiffel Tower built and how tall is it?"
        
        print(f"\n📝 Testing query: {query}")
        
        response = mistral.generate_response(query, context_docs)
        
        print("\n🤖 Response:")
        print(response["answer"])
        print("\n📚 Sources:")
        for source in response["sources"]:
            print(f"- {source}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_rag_system()