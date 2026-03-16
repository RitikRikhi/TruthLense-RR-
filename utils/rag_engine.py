import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chromadb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGEngine:
    def __init__(self):
        print(f"Initializing RAG Vector Database at {CHROMA_DB_PATH}...")
        
        # 1. Initialize the embedding model (used to turn text into math vectors)
        # We use a fast, lightweight SentenceTransformer for hackathons
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize ChromaDB (Local Vector Store)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Create or load the collection where our factual ground-truth will live
        self.fact_collection = self.chroma_client.get_or_create_collection(
            name="truthlens_facts",
            metadata={"hnsw:space": "cosine"} # Use Cosine Similarity for text matching
        )
        
        # 3. Initialize the Generative Model (Gemini acting as our synthesizer)
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.gemini_model = None
            print("Warning: GEMINI_API_KEY not found. RAG generation will be disabled.")
            
    def ingest_facts(self, facts: list[str], sources: list[str]):
        """
        Takes a list of factual text strings, generates their embeddings, 
        and stores them in ChromaDB.
        """
        if not facts:
            return 0
            
        print(f"Embedding and storing {len(facts)} facts into Vector DB...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(facts).tolist()
        
        # Generate IDs
        ids = [f"fact_{i}_{os.urandom(4).hex()}" for i in range(len(facts))]
        
        # Format metadata
        metadatas = [{"source": source} for source in sources]
        
        # Insert into ChromaDB
        self.fact_collection.add(
            embeddings=embeddings,
            documents=facts,
            metadatas=metadatas,
            ids=ids
        )
        
        print("Ingestion complete.")
        return len(facts)
        
    def query_rag(self, user_query: str) -> dict:
        """
        The core Hybrid RAG loop:
        1. Embed user query
        2. Retrieve top K facts
        3. Generate answer using Gemini
        """
        if self.fact_collection.count() == 0:
            return {
                "rag_score": 0,
                "rag_explanation": "Vector Database is empty. Please ingest facts first.",
                "retrieved_context": []
            }
            
        # 1. Embed query
        query_embedding = self.embedder.encode([user_query]).tolist()
        
        # 2. Retrieve Top-3 most mathematically similar facts from our DB
        results = self.fact_collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        retrieved_docs = results['documents'][0]
        context_string = "\n".join([f"- {doc}" for doc in retrieved_docs])
        
        # 3. Generate Analysis combining external context + user query
        if self.gemini_model:
            prompt = f"""
            You are TruthLens, an AI fact-checker. 
            Analyze the user's statement for misinformation.
            
            Retrieved Factual Context from Database:
            {context_string}
            
            User Statement to Verify:
            "{user_query}"
            
            Based ONLY on the retrieved factual context above, is the user's statement misleading?
            Output your response strictly as a JSON object with two keys:
            1. "score": An integer from 0 to 100 (100 = Completely Credible, 0 = Completely False/Misleading)
            2. "explanation": A short 1-sentence string explaining why based on the context.
            """
            
            try:
                response = self.gemini_model.generate_content(prompt)
                
                # Naive JSON extraction (for Hackathon simplicity)
                import json
                # Clean markdown blocks if present
                clean_json = response.text.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_json)
                
                return {
                    "rag_score": int(result.get("score", 50)),
                    "rag_explanation": result.get("explanation", "Generation failed to yield explanation."),
                    "retrieved_context": retrieved_docs
                }
            except Exception as e:
                print(f"Gemini generation error: {e}")
                
        # Fallback if Gemini is broken/unconfigured
        return {
            "rag_score": 50,
            "rag_explanation": f"Found similar facts, but Gemini Generation failed or is disabled.",
            "retrieved_context": retrieved_docs
        }
        
# Singleton pattern for the FastAPI app
_rag_instance = None

def get_rag_engine() -> RAGEngine:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance
