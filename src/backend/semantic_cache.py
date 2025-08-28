import os
import uuid
import re
import json
import datetime
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import spacy
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

@dataclass
class QueryTemplate:
    intent: str
    template: str
    parameters: List[str]
    cypher_template: str
    priority: int = 1
    aliases: List[str] = field(default_factory=list)
    parameter_patterns: Dict[str, str] = field(default_factory=dict)

@dataclass 
class CacheHit:
    cypher_query: str
    response: Optional[str]
    confidence: float
    strategy: str
    template_used: Optional[str] = None

class AdvancedParameterExtractor:
    """Estrattore di parametri più sofisticato usando NLP"""
    
    def __init__(self):
        # Inizializza spaCy per NLP (se disponibile)
        self.nlp = None
        self.use_nlp = False
        
        # Prova diversi modelli in ordine di preferenza
        spacy_models = ["en_core_web_sm", "en_core_web_md", "en"]
        
        for model_name in spacy_models:
            try:
                import spacy
                self.nlp = spacy.load(model_name)
                self.use_nlp = True
                logging.info(f"Loaded spaCy model: {model_name}")
                break
            except (OSError, ImportError) as e:
                logging.debug(f"Could not load spaCy model {model_name}: {e}")
                continue
        
        if not self.use_nlp:
            logging.warning("spaCy not available, using regex-based extraction only")
        
        # Pattern comuni per diversi tipi di parametri
        self.param_patterns = {
            'count': r'\b(?:top|first|last|\d+)\s+(?:\d+)\b',
            'number': r'\b\d+\b',
            'name': r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b',
            'company': r'\b(?:company|corp|inc|ltd)\s+[A-Z][a-zA-Z]+\b',
            'project': r'\b(?:project|proj)\s+[A-Z][a-zA-Z]+\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Estrae entità dal testo usando NLP"""
        entities = defaultdict(list)
        
        if self.use_nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'CARDINAL']:
                    entities[ent.label_.lower()].append(ent.text)
        
        # Fallback con regex
        numbers = re.findall(r'\b\d+\b', text)
        names = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b', text)
        
        if numbers:
            entities['number'].extend(numbers)
        if names:
            entities['person'].extend(names)
            
        return dict(entities)
    
    def extract_parameters_advanced(self, question: str, template: QueryTemplate) -> Dict[str, str]:
        """Estrazione avanzata dei parametri"""
        # Normalizza la domanda
        normalized_q = self._normalize_for_extraction(question)
        normalized_t = self._normalize_for_extraction(template.template)
        
        # Estrattore basato su pattern specifici del template
        if template.parameter_patterns:
            params = {}
            for param_name, pattern in template.parameter_patterns.items():
                matches = re.findall(pattern, normalized_q, re.IGNORECASE)
                if matches:
                    params[param_name] = matches[0] if isinstance(matches[0], str) else matches[0][0]
            if len(params) == len(template.parameters):
                return params
        
        # Fallback su estrazione NLP
        entities = self.extract_entities(question)
        params = {}
        
        for param in template.parameters:
            if param == 'count' and 'number' in entities:
                params[param] = entities['number'][0]
            elif param == 'user' and 'person' in entities:
                params[param] = entities['person'][0]
            elif param == 'company' and 'org' in entities:
                params[param] = entities['org'][0]
            elif param == 'project':
                # Cerca parole dopo "project"
                project_match = re.search(r'project\s+([A-Za-z0-9_]+)', normalized_q, re.IGNORECASE)
                if project_match:
                    params[param] = project_match.group(1)
        
        return params
    
    def _normalize_for_extraction(self, text: str) -> str:
        """Normalizza il testo per l'estrazione"""
        # Rimuovi punteggiatura
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalizza spazi
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class SemanticCache:
    def __init__(self, collection_name: str, mode: str, embedder: str, vector_size: int):
        self.collection_name = collection_name
        self.mode = mode
        self.embedder = embedder
        self.vector_size = vector_size
        
        # Cache più sofisticata con LRU
        self.embedding_cache = OrderedDict()
        self.max_cache_size = 10000
        
        # Statistiche per ottimizzazione
        self.cache_hits = 0
        self.cache_misses = 0
        self.template_hits = 0
        
        # Estrattore parametri avanzato
        self.param_extractor = AdvancedParameterExtractor()
        
        # Template collection
        self.template_collection = f"{collection_name}_templates"
        
        # Template migliorati con pattern e priorità
        self.query_templates = self._load_enhanced_templates()
        
        # Inizializzazione
        self._initialize_qdrant()
        
        # Cache in-memory per query frequenti
        self.frequent_queries_cache = OrderedDict()
        self.max_frequent_cache = 100

    def _load_enhanced_templates(self) -> List[QueryTemplate]:
        """Carica template migliorati con pattern specifici"""
        return [
            QueryTemplate(
                intent="get_top_users_by_project",
                template="get top {count} users from project {project}",
                parameters=["count", "project"],
                cypher_template="MATCH (p:Person)-[:WORK_ON]->(proj:Project {name: '{project}'}) RETURN u LIMIT {count}",
                priority=1,
                aliases=["show first {count} users in project {project}", "list top {count} people from {project}"],
                parameter_patterns={
                    "count": r'\b(?:top|first)\s+(\d+)\b',
                    "project": r'project\s+([A-Za-z0-9_]+)'
                }
            ),
            QueryTemplate(
                intent="list_projects_by_user",
                template="list projects for user {user}",
                parameters=["user"],
                cypher_template="MATCH (p:Person {name: '{user}'})-[:WORK_ON]->(proj:Project) RETURN proj",
                priority=1,
                aliases=["show projects of {user}", "what projects does {user} work on"],
                parameter_patterns={
                    "user": r'(?:for|of|does)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)'
                }
            ),
            QueryTemplate(
                intent="list_user_by_company",
                template="list all people who work for company {company}",
                parameters=["company"],
                cypher_template="MATCH (p:Person)-[:WORKS_FOR]->(c:Company {name: '{company}'}) RETURN p",
                priority=2,
                aliases=["who works at {company}", "all staff from {company}", "find people in {company}", "employees of {company}"],
                parameter_patterns={
                    "company": r'(?:company|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)'
                }
            ),
            QueryTemplate(
                intent="list_project_and_company_by_user",
                template="find projects {user} works on and company he works for",
                parameters=["user"],
                cypher_template="MATCH (p:Person {name: '{user}'})-[:WORK_ON]->(proj:Project) OPTIONAL MATCH (p)-[:WORKS_FOR]->(comp:Company) RETURN proj.title AS Project, comp.name AS Company",
                priority=3,
                aliases=[
                    "what projects is {user} involved in and where does he work",
                    "show {user}'s projects and employer",
                    "which company does {user} work for and what are his projects",
                    "get {user}'s work details",
                    "info on {user}'s projects and company"
                ],
                parameter_patterns={
                    "user": r'(?:find\s+projects\s+|find\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+works'
                }
            ),
            QueryTemplate(
                intent="find_people_on_project_by_date_range",
                template="find all the people working on the project {project} between {start_year} and {end_year}",
                parameters=["project", "start_year", "end_year"],
                cypher_template="MATCH (p:Person)-[w:WORK_ON]->(pr:Project {title: '{project}'}) "
                                "WHERE date(w.start_date) <= date('{end_year}-12-31') "
                                "AND (date(w.end_date) IS NULL OR date(w.end_date) >= date('{start_year}-01-01')) "
                                "RETURN p.name AS PersonName, p.id_person AS PersonID",
                priority=4,
                aliases=[
                    "who worked on project {project} from {start_year} to {end_year}",
                    "list all individuals on {project} between {start_year} and {end_year}",
                    "people working on {project} during {start_year} and {end_year}",
                    "find workers for project {project} between years {start_year} and {end_year}",
                    "show staff on {project} from {start_year} to {end_year}",
                    "get everyone who worked on {project} from {start_year} until {end_year}"
                ],
                parameter_patterns={
                    "project": r"project(?: called)?\s*(?:'([^']+)'|\"([^\"]+)\"|(\b[A-Za-z0-9\s-]+?)(?=\s+(?:between|from|during)))",
                    "start_year": r"(?:between|from|during)\s+(\d{4})",
                    "end_year": r"(?:and|to|until)\s+(\d{4})(?!\s*\d)"
                }
            ),
            QueryTemplate(
                intent="find_people_in_company_by_date_range",
                template="find all the people who worked for {company} between {start_year} and {end_year}",
                parameters=["company", "start_year", "end_year"],
                cypher_template="MATCH (p:Person)-[w:WORKS_FOR]->(c:Company {name: '{company}'}) "
                                "WHERE date(w.start_date) <= date('{end_year}-12-31') "
                                "AND (date(w.end_date) IS NULL OR date(w.end_date) >= date('{start_year}-01-01')) "
                                "RETURN p.name AS PersonName, p.id_person AS PersonID, c.name AS CompanyName",
                priority=4,
                aliases=[
                    "who worked at {company} from {start_year} to {end_year}",
                    "list all employees of {company} between {start_year} and {end_year}",
                    "people employed by {company} during {start_year} and {end_year}",
                    "find staff for {company} between years {start_year} and {end_year}",
                    "show personnel at {company} from {start_year} to {end_year}",
                    "get everyone who was at {company} from {start_year} until {end_year}"
                ],
                parameter_patterns={
                    "company": r"(?:for|at|company(?: called|named)?)\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Z][a-zA-Z\s-]+))(?=\s*(?:between|from|during|\.|\?|$))",
                    "start_year": r"(?:between|from|during)\s+(\d{4})",
                    "end_year": r"(?:and|to|until)\s+(\d{4})(?!\s*\d)"
                }
            ),
        ]

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
            raise ValueError("Qdrant has 3 modes: memory, cloud or docker")
        
        # Create collections with error handling
        self._create_collection_safe(self.collection_name)
        self._create_collection_safe(self.template_collection)
        
        # Initialize embedding model
        self._initialize_embedder()
        
        logging.info("Enhanced Semantic Cache Initialized")

    def _create_collection_safe(self, collection_name: str):
        """Safely create collection with error handling"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logging.info(f"Created collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error creating collection {collection_name}: {e}")
            raise

    def _initialize_embedder(self):
        """Initialize embedding model with fallback"""
        try:
            supported_models = [x["model"] for x in TextEmbedding.list_supported_models()]
            embedder_model = os.environ.get("EMBEDDER", self.embedder)
            
            if embedder_model in supported_models:
                logging.info(f"Using registered model: {embedder_model}")
                self.model = TextEmbedding(model_name=embedder_model)
            else:
                logging.info(f"Registering custom model: {embedder_model}")
                TextEmbedding.add_custom_model(
                    model=embedder_model,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=embedder_model),
                    dim=self.vector_size,
                    model_file="onnx/model.onnx",
                )
                self.model = TextEmbedding(model_name=embedder_model)
        except Exception as e:
            logging.error(f"Error initializing embedder: {e}")
            raise

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with improved LRU caching"""
        if not text or not text.strip():
            return None
            
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Check cache (LRU behavior)
        if text_hash in self.embedding_cache:
            # Move to end (most recently used)
            embedding = self.embedding_cache.pop(text_hash)
            self.embedding_cache[text_hash] = embedding
            self.cache_hits += 1
            return embedding
        
        try:
            result = list(self.model.embed([text]))
            if not result:
                return None
                
            embedding = result[0].tolist() if hasattr(result[0], 'tolist') else result[0]
            
            # Cache management (LRU)
            if len(self.embedding_cache) >= self.max_cache_size:
                self.embedding_cache.popitem(last=False)  # Remove oldest
            
            self.embedding_cache[text_hash] = embedding
            self.cache_misses += 1
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            return None

    def find_best_template_match(self, question: str, threshold: float = 0.75) -> Optional[Tuple[QueryTemplate, Dict[str, str], float]]:
        """Trova il miglior template match con scoring migliorato"""
        normalized_question = self.normalize_question(question)
        
        # Prima prova con cache in-memory per query frequenti
        freq_cache_key = hashlib.md5(normalized_question.encode()).hexdigest()
        if freq_cache_key in self.frequent_queries_cache:
            cached_result = self.frequent_queries_cache.pop(freq_cache_key)
            self.frequent_queries_cache[freq_cache_key] = cached_result  # Move to end
            return cached_result
        
        question_embedding = self.get_embedding(normalized_question)
        if question_embedding is None:
            return None
        
        best_match = None
        best_score = 0.0
        best_params = {}
        
        # Cerca nei template ordinati per priorità
        sorted_templates = sorted(self.query_templates, key=lambda x: x.priority)
        
        for template in sorted_templates:
            # Calcola similarità semantica
            template_embedding = self.get_embedding(template.template)
            if template_embedding is None:
                continue
                
            # Similarità coseno semplificata
            similarity = self._cosine_similarity(question_embedding, template_embedding)
            
            # Bonus per match esatto di parole chiave
            keyword_bonus = self._calculate_keyword_bonus(normalized_question, template.template)
            
            # Prova estrazione parametri
            extracted_params = self.param_extractor.extract_parameters_advanced(question, template)
            
            # Calcola score finale
            param_completeness = len(extracted_params) / len(template.parameters) if template.parameters else 1.0
            final_score = (similarity * 0.6) + (keyword_bonus * 0.2) + (param_completeness * 0.2)
            
            # Bonus priorità
            priority_bonus = (5 - template.priority) * 0.02  # Max 8% bonus
            final_score += priority_bonus
            
            if final_score > best_score and final_score >= threshold and len(extracted_params) == len(template.parameters):
                best_score = final_score
                best_match = template
                best_params = extracted_params
        
        result = (best_match, best_params, best_score) if best_match else None
        
        # Cache result se valido
        if result and len(self.frequent_queries_cache) < self.max_frequent_cache:
            self.frequent_queries_cache[freq_cache_key] = result
        
        if best_match:
            self.template_hits += 1
            
        return result

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcola similarità coseno"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
                
            return dot_product / (magnitude1 * magnitude2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _calculate_keyword_bonus(self, question: str, template: str) -> float:
        """Calcola bonus per parole chiave corrispondenti"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        template_words = set(re.findall(r'\b\w+\b', template.lower()))
        
        # Rimuovi parole comuni e placeholder
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= stop_words
        template_words -= stop_words
        template_words = {w for w in template_words if not (w.startswith('{') and w.endswith('}'))}
        
        if not template_words:
            return 0.0
            
        intersection = question_words & template_words
        return len(intersection) / len(template_words)

    def normalize_question(self, question: str) -> str:
        """Normalizzazione avanzata delle domande"""
        if not question:
            return ""
            
        # Converti in minuscolo e rimuovi spazi extra
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        
        # Sostituzioni più complete
        replacements = [
            (r'\bfirst\s+(\d+)\b', r'top \1'),
            (r'\bget\s+me\b', 'get'),
            (r'\bshow\s+me\b', 'show'),
            (r'\blist\s+all\s+the\b', 'list'),
            (r'\blist\s+all\b', 'list'),
            (r'\bwho\s+are\s+the\b', 'list'),
            (r'\bwhat\s+are\s+the\b', 'list'),
            (r'\btell\s+me\s+about\b', 'get'),
            (r'\bfind\s+out\b', 'find'),
            (r'\bi\s+want\s+to\s+know\b', 'get'),
            (r'\bcan\s+you\s+tell\s+me\b', 'get'),
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Rimuovi punteggiatura finale
        normalized = re.sub(r'[.!?]+$', '', normalized)
        
        return normalized

    def smart_search(self, question: str, template_threshold: float = 0.8, exact_threshold: float = 0.95, similarity_threshold: float = 0.8) -> CacheHit:
        """Ricerca intelligente ottimizzata per ridurre uso LLM"""
        if not question or not question.strip():
            return CacheHit("", None, 0.0, "invalid_input")
        
        # Strategy 1: Template matching
        template_match = self.find_best_template_match(question, threshold=template_threshold)
        if template_match:
            template, parameters, confidence = template_match
            cypher_query = self.generate_cypher_from_template(template, parameters)
            
            # Cerca response cached per questa query specifica
            cached_response = self._search_cached_response(cypher_query)
            
            return CacheHit(
                cypher_query=cypher_query,
                response=cached_response,
                confidence=confidence,
                strategy="template_match",
                template_used=template.intent
            )
        
        # Strategy 2: Exact cache hit
        normalized_question = self.normalize_question(question)
        question_embedding = self.get_embedding(normalized_question)
        
        if question_embedding is None:
            return CacheHit("", None, 0.0, "embedding_failed")
        
        exact_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=1,
            score_threshold=exact_threshold,
        ).points
        
        if exact_matches:
            match = exact_matches[0]
            self._update_usage_count_efficient(match.id)
            return CacheHit(
                cypher_query=match.payload["cypher_query"],
                response=match.payload.get("response"),
                confidence=match.score,
                strategy="exact_match"
            )
        
        # Strategy 3: Semantic similarity (soglia più bassa)
        similar_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=3,
            score_threshold=similarity_threshold,
        ).points
        
        if similar_matches:
            best_match = similar_matches[0]
            return CacheHit(
                cypher_query=best_match.payload["cypher_query"],
                response=best_match.payload.get("response"),
                confidence=best_match.score * 0.9,  # Penalizza lievemente
                strategy="semantic_similarity"
            )
        
        return CacheHit("", None, 0.0, "no_match")

    def generate_cypher_from_template(self, template: QueryTemplate, parameters: Dict[str, str]) -> str:
        """Genera query Cypher da template con validazione"""
        if not template.cypher_template:
            raise ValueError(f"Template {template.intent} has no cypher_template")
        
        cypher_query = template.cypher_template
        
        # Sostituisci parametri con escape per sicurezza
        for param, value in parameters.items():
            # Sanitizza il valore (rimuovi caratteri pericolosi)
            safe_value = re.sub(r'[^\w\s-]', '', str(value))
            cypher_query = cypher_query.replace(f"{{{param}}}", safe_value)
        
        return cypher_query

    def _search_cached_response(self, cypher_query: str) -> Optional[str]:
        """Cerca una response cached per una specifica query Cypher"""
        query_hash = hashlib.md5(cypher_query.encode()).hexdigest()
        
        # Cerca nei punti esistenti
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="cypher_hash",
                    match=MatchValue(value=query_hash)
                )
            ]
        )
        
        try:
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=True
            )[0]
            
            if results:
                return results[0].payload.get("response")
        except Exception as e:
            logging.debug(f"Error searching cached response: {e}")
        
        return None

    def store_query_and_response(self, question: str, cypher_query: str, response: str, template_used: Optional[str] = None) -> bool:
        """Store query con metadati migliorati"""
        if not all([question, cypher_query]):
            logging.warning("Invalid input for storing query")
            return False
            
        try:
            question_embedding = self.get_embedding(question)
            if question_embedding is None:
                logging.error("Failed to generate embedding for question")
                return False
            
            normalized_question = self.normalize_question(question)
            cypher_hash = hashlib.md5(cypher_query.encode()).hexdigest()
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=question_embedding,
                payload={
                    "question": question,
                    "normalized_question": normalized_question,
                    "cypher_query": cypher_query,
                    "cypher_hash": cypher_hash,
                    "response": response,
                    "template_used": template_used,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "usage_count": 1,
                    "response_length": len(response) if response else 0
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

    def _update_usage_count_efficient(self, point_id: str):
        """Aggiorna contatore uso in modo efficiente"""
        try:
            # Per ora incrementiamo un contatore locale
            # In futuro si potrebbe implementare un batch update
            pass
        except Exception as e:
            logging.debug(f"Error updating usage count: {e}")

    def get_performance_stats(self) -> Dict:
        """Statistiche performance dettagliate"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                "cache_performance": {
                    "embedding_cache_hits": self.cache_hits,
                    "embedding_cache_misses": self.cache_misses,
                    "hit_rate_percentage": round(cache_hit_rate, 2),
                    "template_hits": self.template_hits
                },
                "storage": {
                    "total_cached_queries": collection_info.points_count,
                    "embedding_cache_size": len(self.embedding_cache),
                    "frequent_queries_cache_size": len(self.frequent_queries_cache),
                    "vector_size": self.vector_size
                },
                "model_info": {
                    "embedder_model": self.embedder,
                    "nlp_enabled": self.param_extractor.use_nlp
                },
                "efficiency": {
                    "templates_loaded": len(self.query_templates),
                    "avg_template_priority": sum(t.priority for t in self.query_templates) / len(self.query_templates)
                }
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}

    def clear_caches(self):
        """Pulisci tutte le cache in-memory"""
        self.embedding_cache.clear()
        self.frequent_queries_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.template_hits = 0
        logging.info("All caches cleared")

# Funzione di utilità per inizializzazione rapida
def create_optimized_cache(collection_name: str = "semantic_cache", **kwargs) -> SemanticCache:
    """Crea una cache semantica ottimizzata con configurazione di default"""
    defaults = {
        "mode": os.getenv("QDRANT_MODE", "memory"),
        "embedder": os.getenv("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_size": int(os.getenv("VECTOR_SIZE", "384"))
    }
    defaults.update(kwargs)
    
    return SemanticCache(collection_name, **defaults)


# import logging
# def run():
#     # Attiva logging per debug
#     logging.basicConfig(level=logging.INFO)

#     # Inizializza la cache in modalità memoria (non serve Qdrant esterno)
#     cache = create_optimized_cache("test_cache", mode="memory")

#     # Domande di esempio
#     questions = [
#         "Get top 5 users from project Apollo",
#         "List projects for user Alice",
#         "Which projects is Bob working on?",
#         "Find projects Charlie works on and company he works for"
#     ]

#     for q in questions:
#         hit = cache.smart_search(q)
#         print(f"\n❓ Question: {q}")
#         print(f"🔎 Strategy: {hit.strategy}")
#         print(f"📊 Confidence: {hit.confidence:.2f}")
#         print(f"📜 Cypher query: {hit.cypher_query}")
#         print(f"💾 Cached response: {hit.response}")

#         # Se vuoi testare anche la memorizzazione
#         if hit.cypher_query:
#             cache.store_query_and_response(
#                 question=q,
#                 cypher_query=hit.cypher_query,
#                 response=f"Dummy response for: {q}",
#                 template_used=hit.template_used,
#             )

#     # Mostra statistiche
#     stats = cache.get_performance_stats()
#     print("\n📈 Performance stats:", stats)

# if __name__ == "__main__":
#     run()
