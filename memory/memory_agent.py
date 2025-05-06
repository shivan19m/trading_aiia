import chromadb
from chromadb.utils import embedding_functions
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryAgent:
    """
    Stores immutable plan history, performance logs, constraint violation records, and vector memory for plan retrieval.
    TODO: Integrate with persistent storage or LLM-based memory.
    """
    def __init__(self):
        self.plan_logs = []
        self.performance_logs = []
        self.constraint_violations = []
        # Setup ChromaDB in-memory collection for vector memory
        self.chroma_client = chromadb.Client()
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="text-embedding-ada-002"
        )
        try:
            self.collection = self.chroma_client.create_collection(
                name="plan_vectors",
                metadata={"hnsw:space": "cosine"}
            )
        except chromadb.errors.InternalError as e:
            if "already exists" in str(e):
                # If collection exists, get it instead of creating
                self.collection = self.chroma_client.get_collection(
                    name="plan_vectors"
                )
            else:
                raise e

    def store_plan(self, plan, status):
        """
        Store plan and its execution status.
        """
        self.plan_logs.append({'plan': plan, 'status': status})

    def log_performance(self, plan, result):
        """
        Store performance outcome for a plan.
        """
        self.performance_logs.append({'plan': plan, 'result': result})

    def record_violation(self, plan, violation):
        """
        Store constraint violation record.
        """
        self.constraint_violations.append({'plan': plan, 'violation': violation})

    def get_plan_history(self):
        return self.plan_logs

    def get_performance_history(self):
        return self.performance_logs

    def get_constraint_violations(self):
        return self.constraint_violations

    def store_plan_vector(self, plan, context):
        """
        Store plan in vector memory for similarity search.
        """
        try:
            # Convert plan to string for embedding
            plan_str = str(plan)
            # Store in ChromaDB
            self.collection.add(
                documents=[plan_str],
                metadatas=[{"context": context}],
                ids=[f"plan_{len(self.plan_logs)}"]
            )
            logger.debug(f"Stored plan vector: {plan_str[:100]}...")
        except Exception as e:
            logger.error(f"Failed to store plan vector: {str(e)}")

    def retrieve_similar_plans(self, context, top_k=3):
        """
        Retrieve similar plans from vector memory based on context.
        """
        try:
            results = self.collection.query(
                query_texts=[context],
                n_results=top_k
            )
            logger.debug(f"Retrieved {len(results['documents'][0])} similar plans")
            return results['documents'][0]
        except Exception as e:
            logger.error(f"Failed to retrieve similar plans: {str(e)}")
            return [] 