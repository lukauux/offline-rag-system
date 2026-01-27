"""
Demo script for Enhanced RAG Pipeline
Shows how to use the system with sample data.
"""

import os
import json
from pathlib import Path
from enhanced_rag_pipeline import EnhancedRAGPipeline

def demo_rag_system():
    """Demonstrate the enhanced RAG system"""
    
    print("="*70)
    print("🚀 Enhanced Multimodal RAG System Demo")
    print("="*70)
    
    # Initialize pipeline
    print("\n[Step 1] Initializing RAG Pipeline...")
    pipeline = EnhancedRAGPipeline()
    
    # Show initial stats
    print("\n[Step 2] System Statistics:")
    stats = pipeline.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Check for sample data
    sample_dir = Path(".data/sample_documents")
    
    if sample_dir.exists() and any(sample_dir.iterdir()):
        print(f"\n[Step 3] Ingesting files from {sample_dir}...")
        result = pipeline.ingest_directory(str(sample_dir))
        print(f"\n✅ Ingestion Results: {result}")
    else:
        print(f"\n[Step 3] No sample documents found in {sample_dir}")
        print("ℹ️  Add documents to .data/sample_documents/ to test ingestion")
    
    # Demo queries
    print("\n[Step 4] Running Demo Queries...")
    
    demo_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What are neural networks?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Query {i} ---")
        
        # Try different search strategies
        print(f"\n🔍 Standard Search:")
        result = pipeline.query(query, search_type="similarity", k=3)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['num_sources']}")
        
        print(f"\n🔍 MMR Search (diverse results):")
        result = pipeline.query(query, search_type="mmr", k=3)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['num_sources']}")
        
        print(f"\n🔍 Multi-Query Search:")
        result = pipeline.query(query, use_multi_query=True, k=3)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['num_sources']}")
    
    # Save state
    print("\n[Step 5] Saving system state...")
    pipeline.save_state()
    
    # Final stats
    print("\n[Step 6] Final Statistics:")
    final_stats = pipeline.get_stats()
    print(json.dumps(final_stats, indent=2))
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    
    print("\n📚 Next Steps:")
    print("1. Add your documents to .data/sample_documents/")
    print("2. Run: python scripts/demo_enhanced_rag.py")
    print("3. Use the web UI for interactive queries")

if __name__ == "__main__":
    demo_rag_system()
