import os
import pandas as pd
from utils.rag_engine import get_rag_engine
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "./data/training_data.csv")

def ingest_from_csv(file_path: str):
    """
    Reads a CSV dataset containing known facts, extracts the text,
    and ingests it into the ChromaDB RAG Vector Store.
    
    Expected CSV columns:
    1. 'text' - The actual factual content or news headline
    2. 'source' (optional) - The news outlet or source URL
    """
    
    print(f"Loading factual dataset from {file_path} for RAG Ingestion...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please provide a CSV file with a 'text' column.")
        
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
        
    # Optional 'source' for metadata citation in RAG
    sources = df['source'].tolist() if 'source' in df.columns else ["Unknown CSV Source"] * len(df)
    facts = df['text'].astype(str).tolist()
    
    print(f"Retrieved {len(facts)} facts from dataset.")
    
    # Send to the RAG Engine for embedding and database insertion
    engine = get_rag_engine()
    total_added = engine.ingest_facts(facts, sources)
    
    print(f"Successfully ingested {total_added} facts into ChromaDB.")
    return total_added

if __name__ == "__main__":
    # Test script - expects a dummy data file
    import os
    os.makedirs("./data", exist_ok=True)
    
    test_csv = "./data/rag_test_facts.csv"
    if not os.path.exists(test_csv):
        print(f"Creating a sample factual dataset at {test_csv}...")
        pd.DataFrame({
            "text": [
                "The Eiffel Tower is located in Paris, France.",
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "Python was created by Guido van Rossum and first released in 1991.",
                "The speed of light in a vacuum is approximately 299,792 kilometers per second."
            ],
            "source": [
                "Geography Encyclopedia",
                "Physics Text",
                "Programming History",
                "Physics Text"
            ]
        }).to_csv(test_csv, index=False)
        
    ingest_from_csv(test_csv)
