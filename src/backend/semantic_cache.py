import os
import uuid
import re
import json
import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
import hashlib
import logging

@dataclass
class QueryTemplate:
    intent: str
    template: str
    parameters: List[str]
    cypher_template: str

class SemanticCache:
    def __init__(
        self, collection_name: str, mode: str, embedder: str, vector_size: int
    ):
        self.collection_name = collection_name
        self.mode = mode
        self.embedder = embedder
        self.vector_size = vector_size
        self.embedding_cache = {}
        self.max_cache_size = 10000
        
        # Template collection for storing query patterns
        self.template_collection = f"{collection_name}_templates"
        
        # Common query templates (puoi estendere questo)
        self.query_templates = [
            QueryTemplate(
                intent="get_top_users_by_project",
                template="get top {count} users from project {project}",
                parameters=["count", "project"],
                cypher_template="MATCH (u:User)-[:WORKS_ON]->(p:Project {{name: '{project}'}}) RETURN u LIMIT {count}"
            ),
            QueryTemplate(
                intent="list_projects_by_user",
                template="list projects for user {user}",
                parameters=["user"],
                cypher_template="MATCH (u:User {{name: '{user}'}})-[:WORKS_ON]->(p:Project) RETURN p"
            ),
            QueryTemplate(
                intent="list_user_by_company",
                template="List all the people who work for the company {company}.",
                parameters=["company"],
                cypher_template="MATCH (p:Person)-[:WORKS_FOR]->(c:Company {{name: '{company}'}}) RETURN p"
            ),
            QueryTemplate(
                intent="list_project_user_working_on",
                template="What projects is {user} working on?",
                parameters=["user"],
                cypher_template="MATCH (p:Person {{name: '{user}'}})-[:WORK_ON]->(proj:Project) RETURN proj"
            ),
            QueryTemplate(
                intent="list_project_and_company_by_user",
                template="Find the projects {user} works on and what company he works for.",
                parameters=["user"],
                cypher_template="MATCH (p:Person {{name: '{user}'}})-[:WORK_ON]->(proj:Project) OPTIONAL MATCH (p)-[:WORKS_FOR]->(comp:Company) RETURN proj.title AS Project, comp.name AS Company"
            ),
        ]

        self._initialize_qdrant()
        self._initialize_templates()

    def _initialize_qdrant(self):
        """Initialize Qdrant client and collections"""
        if os.getenv("QDRANT_MODE") == "memory":
            self.qdrant_client = QdrantClient(":memory:")
        elif os.getenv("QDRANT_MODE") == "cloud":
            self.qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
        elif os.getenv("QDRANT_MODE") == "docker":
            self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
        else:
            raise ValueError("Qdrant has 3 mode: memory, cloud or docker")
        
        # Create main collection
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        
        # Create template collection
        if not self.qdrant_client.collection_exists(self.template_collection):
            self.qdrant_client.create_collection(
                collection_name=self.template_collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        
        # Initialize embedding model
        supported_models = [x["model"] for x in TextEmbedding.list_supported_models()]
        embedder_model = os.environ.get("EMBEDDER")
        if embedder_model in supported_models:
            print(f"The model {embedder_model} is already registered.")
            self.model = TextEmbedding(model_name=self.embedder)
        else:
            print(f"The model {embedder_model} isn't already registered.")
            TextEmbedding.add_custom_model(
                model=self.embedder,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=self.embedder),
                dim=self.vector_size,
                model_file="onnx/model.onnx",
            )
            self.model = TextEmbedding(model_name=self.embedder)
        
        print("Enhanced Semantic Cache Initialized")

    def _initialize_templates(self):
        """Store query templates in the template collection"""
        for template in self.query_templates:
            template_embedding = self.get_embedding(template.template)
            if template_embedding is None:
                continue
                
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=template_embedding,
                payload={
                    "intent": template.intent,
                    "template": template.template,
                    "parameters": template.parameters,
                    "cypher_template": template.cypher_template,
                    "type": "template"
                },
            )
            self.qdrant_client.upsert(
                collection_name=self.template_collection,
                wait=True,
                points=[point],
            )

    def get_embedding(self, text: str):
        """Get embedding with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            result = list(self.model.embed([text]))
            embedding = result[0]
            
            # Cache management
            if len(self.embedding_cache) >= self.max_cache_size:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            return None

    def extract_parameters(self, question: str, template: str) -> Dict[str, str]:
        """Extract parameters from question using template matching"""
        # Convert template to regex pattern
        pattern = template
        param_positions = {}
        
        # Find parameters in template and create regex pattern
        import re
        param_regex = r'\{(\w+)\}'
        params = re.findall(param_regex, template)
        
        # Create regex pattern by replacing parameters with capture groups
        for param in params:
            pattern = pattern.replace(f"{{{param}}}", r'([^,\s]+)')
        
        # Try to match the question against the pattern
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            extracted_params = {}
            for i, param in enumerate(params):
                if i < len(match.groups()):
                    extracted_params[param] = match.group(i + 1)
            return extracted_params
        
        return {}

    def normalize_question(self, question: str) -> str:
        """Normalize question for better matching"""
        # Convert to lowercase
        normalized = question.lower().strip()
        
        # Replace common variations
        replacements = {
            r'\bfirst\s+(\d+)\b': r'top \1',
            r'\bget\s+me\b': 'get',
            r'\bshow\s+me\b': 'show',
            r'\blist\s+all\b': 'list',
        }
        
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized

    def find_template_match(self, question: str, threshold: float = 0.8) -> Optional[Tuple[QueryTemplate, Dict[str, str]]]:
        """Find matching template for a question"""
        normalized_question = self.normalize_question(question)
        question_embedding = self.get_embedding(normalized_question)
        
        if question_embedding is None:
            return None
        
        # Search in template collection
        search_results = self.qdrant_client.query_points(
            collection_name=self.template_collection,
            query=question_embedding,
            limit=3,
            score_threshold=threshold,
        ).points
        
        for result in search_results:
            payload = result.payload
            template = QueryTemplate(
                intent=payload["intent"],
                template=payload["template"],
                parameters=payload["parameters"],
                cypher_template=payload["cypher_template"]
            )
            
            # Try to extract parameters
            parameters = self.extract_parameters(normalized_question, payload["template"])
            
            if parameters:  # Se riusciamo ad estrarre i parametri
                return template, parameters
        
        return None

    def generate_cypher_from_template(self, template: QueryTemplate, parameters: Dict[str, str]) -> str:
        """Generate Cypher query from template and parameters"""
        cypher_query = template.cypher_template
        
        for param, value in parameters.items():
            cypher_query = cypher_query.replace(f"{{{param}}}", value)
        
        return cypher_query

    def store_query_and_response(self, question: str, cypher_query: str, response: str, template_used: Optional[str] = None):
        """Store query with enhanced metadata"""
        try:
            question_embedding = self.get_embedding(question)
            if question_embedding is None:
                logging.error("Failed to generate embedding for question")
                return False
            
            # Create semantic signature for similar query detection
            normalized_question = self.normalize_question(question)
            semantic_signature = self._create_semantic_signature(normalized_question)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=question_embedding,
                payload={
                    "question": question,
                    "normalized_question": normalized_question,
                    "semantic_signature": semantic_signature,
                    "cypher_query": cypher_query,
                    "response": response,
                    "template_used": template_used,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "usage_count": 1
                },
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point],
            )
            
            logging.info(f"Successfully stored query: {question[:50]}...")
            return True
        except Exception as e:
            logging.error(f"Error storing query and response: {e}")
            return False

    def _create_semantic_signature(self, normalized_question: str) -> str:
        """Create a semantic signature for grouping similar queries"""
        # Extract key concepts (nouns, verbs, numbers)
        import re
        
        # Extract numbers
        numbers = re.findall(r'\d+', normalized_question)
        
        # Extract key words (simplified approach)
        key_words = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'me', 'get', 'show'}
        words = re.findall(r'\b\w+\b', normalized_question)
        
        for word in words:
            if word.lower() not in stop_words and len(word) > 2:
                key_words.append(word.lower())
        
        # Create signature
        signature_parts = []
        signature_parts.extend(sorted(key_words))
        signature_parts.append(f"num_count_{len(numbers)}")
        
        return "_".join(signature_parts[:5])  # Limit to first 5 components

    def smart_search(self, question: str) -> Dict:
        """Enhanced search with multiple strategies"""
        result = {
            "found_exact": False,
            "found_template": False,
            "found_similar": False,
            "cypher_query": None,
            "response": None,
            "confidence": 0.0,
            "strategy_used": None
        }
        
        # Strategy 1: Check for template match first
        template_match = self.find_template_match(question)
        if template_match:
            template, parameters = template_match
            result["cypher_query"] = self.generate_cypher_from_template(template, parameters)
            result["found_template"] = True
            result["strategy_used"] = "template"
            result["confidence"] = 0.9
            
            # Optionally, try to find a cached result with these parameters
            cached_result = self.search_by_semantic_signature(question)
            if cached_result:
                result["response"] = cached_result["response"]
                result["found_exact"] = True
                result["confidence"] = 1.0
            
            return result
        
        # Strategy 2: Exact semantic search
        normalized_question = self.normalize_question(question)
        question_embedding = self.get_embedding(normalized_question)
        
        if question_embedding is None:
            return result
        
        # High threshold search for exact matches
        exact_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=1,
            score_threshold=0.98,
        ).points
        
        if exact_matches:
            match = exact_matches[0]
            result.update({
                "found_exact": True,
                "cypher_query": match.payload["cypher_query"],
                "response": match.payload["response"],
                "confidence": match.score,
                "strategy_used": "exact_match"
            })
            
            # Update usage count
            self._update_usage_count(match.id)
            return result
        
        # Strategy 3: Semantic signature search
        signature_result = self.search_by_semantic_signature(question)
        if signature_result:
            result.update({
                "found_similar": True,
                "cypher_query": signature_result["cypher_query"],
                "response": signature_result["response"],
                "confidence": 0.8,
                "strategy_used": "semantic_signature"
            })
            return result
        
        # Strategy 4: Lower threshold semantic search
        similar_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=3,
            score_threshold=0.85,
        ).points
        
        if similar_matches:
            best_match = similar_matches[0]
            result.update({
                "found_similar": True,
                "cypher_query": best_match.payload["cypher_query"],
                "response": best_match.payload.get("response"),
                "confidence": best_match.score,
                "strategy_used": "semantic_similarity"
            })
        
        return result

    def search_by_semantic_signature(self, question: str) -> Optional[Dict]:
        """Search by semantic signature for parameter-agnostic matching"""
        normalized_question = self.normalize_question(question)
        signature = self._create_semantic_signature(normalized_question)
        
        # Search for queries with similar semantic signatures
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="semantic_signature",
                    match=MatchValue(value=signature)
                )
            ]
        )
        
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self.get_embedding(normalized_question),
            query_filter=filter_condition,
            limit=1,
            score_threshold=0.7
        )
        
        if results:
            return {
                "cypher_query": results[0].payload["cypher_query"],
                "response": results[0].payload.get("response"),
                "confidence": results[0].score
            }
        
        return None

    def _update_usage_count(self, point_id: str):
        """Update usage count for a cached query"""
        try:
            # This would require fetching the point, updating the payload, and upserting
            # For now, we'll skip this implementation detail
            pass
        except Exception as e:
            logging.error(f"Error updating usage count: {e}")

    def get_cache_stats(self) -> Dict:
        """Get enhanced cache statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            template_info = self.qdrant_client.get_collection(self.template_collection)
            
            # Get sample of points
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze semantic signatures
            signatures = [p.payload.get("semantic_signature", "") for p in points if p.payload.get("semantic_signature")]
            signature_counts = {}
            for sig in signatures:
                signature_counts[sig] = signature_counts.get(sig, 0) + 1
            
            stats = {
                "total_cached_queries": collection_info.points_count,
                "total_templates": template_info.points_count,
                "vector_size": self.vector_size,
                "embedding_cache_size": len(self.embedding_cache),
                "top_semantic_signatures": sorted(signature_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "sample_questions": [p.payload.get("question", "")[:100] for p in points[:5]],
                "embedder_model": self.embedder
            }
            
            return stats
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}