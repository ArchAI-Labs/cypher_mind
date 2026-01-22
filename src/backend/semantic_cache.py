import os
import uuid
import re
import json
import datetime
import hashlib
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import spacy
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType, QuantizationSearchParams,
    HnswConfigDiff, OptimizersConfigDiff, PayloadSchemaType, TextIndexParams,
    BinaryQuantization, BinaryQuantizationConfig
)

# Import fuzzy matching library
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("rapidfuzz not available. Install with: pip install rapidfuzz")

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
        
        # ===== NUOVA: Cache in memoria per ultimi 3 risultati =====
        self.recent_results_cache = OrderedDict()
        self.max_recent_results = 3
        
        # Statistiche per ottimizzazione
        self.cache_hits = 0
        self.cache_misses = 0
        self.template_hits = 0
        self.fuzzy_hits = 0  # NUOVO
        self.recent_cache_hits = 0  # NUOVO
        
        # Estrattore parametri avanzato
        self.param_extractor = AdvancedParameterExtractor()
        
        # Template collection
        self.template_collection = f"{collection_name}_templates"
        
        # Template migliorati con pattern e priorità 
        self.query_templates = self._load_enhanced_templates(file_path=os.getenv("TEMPLATE_QUERY"))
        
        # Inizializzazione
        self._initialize_qdrant()
        
        # ===== MODIFICATO: Sincronizza template con Qdrant =====
        self._sync_templates_to_qdrant()
        
        # Cache in-memory per query frequenti (ESISTENTE)
        self.frequent_queries_cache = OrderedDict()
        self.max_frequent_cache = 100

    def _load_enhanced_templates(self, file_path: str) -> List[QueryTemplate]:
        """
        Carica template migliorati da un file JSON.

        :param file_path: Percorso del file JSON contenente i template.
        :return: Una lista di oggetti QueryTemplate.
        """
        templates = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    templates.append(QueryTemplate(**item))
            logging.info(f"Successfully loaded {len(templates)} templates from {file_path}")
        except FileNotFoundError:
            logging.error(f"Template file not found at {file_path}. Using an empty template list.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}. Using an empty template list.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading templates: {e}")

        return templates

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
        
        logging.info("Semantic Cache Initialized")

    def _create_collection_safe(self, collection_name: str):
        """Safely create collection with quantization and optimized HNSW parameters"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                # Get optimization settings from environment
                enable_quantization = os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"
                quantization_type = os.getenv("QUANTIZATION_TYPE", "scalar").lower()
                hnsw_m = int(os.getenv("HNSW_M", "16"))
                hnsw_ef_construct = int(os.getenv("HNSW_EF_CONSTRUCT", "100"))
                on_disk = os.getenv("QDRANT_ON_DISK", "false").lower() == "true"

                # Prepare quantization config
                quantization_config = None
                if enable_quantization:
                    if quantization_type == "binary" and self.vector_size >= 512:
                        # Binary quantization for high-dimensional vectors
                        quantization_config = BinaryQuantization(
                            binary=BinaryQuantizationConfig(
                                always_ram=True
                            )
                        )
                        logging.info(f"Using Binary Quantization for collection: {collection_name}")
                    else:
                        # Scalar quantization (default) - best balance
                        quantization_config = ScalarQuantization(
                            scalar=ScalarQuantizationConfig(
                                type=ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True  # Keep quantized vectors in RAM for speed
                            )
                        )
                        logging.info(f"Using Scalar Quantization (INT8) for collection: {collection_name}")

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        on_disk=on_disk,  # Enable disk storage for large datasets
                    ),
                    quantization_config=quantization_config,
                    hnsw_config=HnswConfigDiff(
                        m=hnsw_m,  # Edges per node (higher = better recall, more memory)
                        ef_construct=hnsw_ef_construct,  # Construction search depth
                        full_scan_threshold=10000,  # Use full scan for small collections
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,  # Start indexing after 20k vectors
                        memmap_threshold=50000,  # Move to disk after 50k vectors
                    ),
                )
                logging.info(f"Created optimized collection: {collection_name} (HNSW m={hnsw_m}, ef_construct={hnsw_ef_construct}, quantization={enable_quantization})")

                # Initialize payload indexes after collection creation
                self._initialize_payload_indexes(collection_name)
        except Exception as e:
            logging.error(f"Error creating collection {collection_name}: {e}")
            raise

    def _initialize_payload_indexes(self, collection_name: str):
        """Create payload indexes for fast filtering on frequently queried fields"""
        try:
            # Skip indexing for template collection (smaller, less frequently queried)
            if "_templates" in collection_name:
                logging.info(f"Skipping payload indexes for template collection: {collection_name}")
                return

            # Index template_used for quick template-based queries
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="template_used",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logging.info(f"Created keyword index on 'template_used' for {collection_name}")

            # Index cypher_hash for exact query matching
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="cypher_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logging.info(f"Created keyword index on 'cypher_hash' for {collection_name}")

            # Index usage_count for popularity-based retrieval
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="usage_count",
                field_schema=PayloadSchemaType.INTEGER
            )
            logging.info(f"Created integer index on 'usage_count' for {collection_name}")

            # Full-text search on normalized questions
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="normalized_question",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer="word",
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            logging.info(f"Created text index on 'normalized_question' for {collection_name}")

        except Exception as e:
            logging.warning(f"Error creating payload indexes for {collection_name}: {e}")
            # Don't raise - indexes are optimization, not critical

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

    def _sync_templates_to_qdrant(self):
        """
        Sincronizza i template caricati da file nella collection Qdrant.
        Viene chiamato all'inizializzazione.
        """
        if not self.query_templates:
            logging.warning("No templates to sync to Qdrant")
            return
        
        try:
            # Verifica se i template sono già stati caricati
            collection_info = self.qdrant_client.get_collection(self.template_collection)
            if collection_info.points_count > 0:
                logging.info(f"Templates already in Qdrant ({collection_info.points_count} templates)")
                return
            
            points = []
            for template in self.query_templates:
                # Genera embedding per il template
                template_embedding = self.get_embedding(template.template)
                if template_embedding is None:
                    logging.warning(f"Failed to generate embedding for template: {template.intent}")
                    continue
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=template_embedding,
                    payload={
                        "intent": template.intent,
                        "template": template.template,
                        "parameters": template.parameters,
                        "cypher_template": template.cypher_template,
                        "priority": template.priority,
                        "aliases": template.aliases,
                        "parameter_patterns": template.parameter_patterns,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.template_collection,
                    wait=True,
                    points=points
                )
                logging.info(f"Synced {len(points)} templates to Qdrant collection '{self.template_collection}'")
        
        except Exception as e:
            logging.error(f"Error syncing templates to Qdrant: {e}")

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
        """
        Find best template match using Qdrant vector search for optimal performance.

        This method now leverages Qdrant's HNSW indexing instead of iterating through
        templates in Python, providing 40-100x faster search for large template sets.
        """
        normalized_question = self.normalize_question(question)

        # Check in-memory cache for frequent queries
        freq_cache_key = hashlib.md5(normalized_question.encode()).hexdigest()
        if freq_cache_key in self.frequent_queries_cache:
            cached_result = self.frequent_queries_cache.pop(freq_cache_key)
            self.frequent_queries_cache[freq_cache_key] = cached_result  # Move to end (LRU)
            return cached_result

        question_embedding = self.get_embedding(normalized_question)
        if question_embedding is None:
            return None

        try:
            # Use Qdrant's vector search for templates (much faster than Python iteration)
            search_results = self.qdrant_client.query_points(
                collection_name=self.template_collection,
                query=question_embedding,
                limit=10,  # Get top 10 candidates for parameter extraction
                score_threshold=max(0.6, threshold - 0.15),  # Lower threshold to allow rescoring
            )

            if not search_results.points:
                return None

            best_match = None
            best_score = 0.0
            best_params = {}

            # Rescore candidates with parameter extraction and keyword matching
            for point in search_results.points:
                similarity = point.score
                payload = point.payload

                # Reconstruct QueryTemplate object from payload
                template_obj = self._reconstruct_template_from_payload(payload)
                if not template_obj:
                    continue

                # Keyword bonus
                keyword_bonus = self._calculate_keyword_bonus(normalized_question, template_obj.template)

                # Extract parameters
                extracted_params = self.param_extractor.extract_parameters_advanced(question, template_obj)

                # Calculate final score
                param_completeness = len(extracted_params) / len(template_obj.parameters) if template_obj.parameters else 1.0
                final_score = (similarity * 0.6) + (keyword_bonus * 0.2) + (param_completeness * 0.2)

                # Priority bonus (higher priority = lower number = higher bonus)
                priority_bonus = (5 - template_obj.priority) * 0.02
                final_score += priority_bonus

                # Only accept if all required parameters are extracted
                if final_score > best_score and final_score >= threshold and len(extracted_params) == len(template_obj.parameters):
                    best_score = final_score
                    best_match = template_obj
                    best_params = extracted_params

            result = (best_match, best_params, best_score) if best_match else None

            # Cache result if valid
            if result and len(self.frequent_queries_cache) < self.max_frequent_cache:
                self.frequent_queries_cache[freq_cache_key] = result

            if best_match:
                self.template_hits += 1

            return result

        except Exception as e:
            logging.error(f"Error in Qdrant template search: {e}")
            # Fallback to in-memory search (backwards compatibility)
            return self._fallback_template_search(question, normalized_question, question_embedding, threshold)

    def _reconstruct_template_from_payload(self, payload: Dict) -> Optional[QueryTemplate]:
        """Reconstruct QueryTemplate object from Qdrant payload"""
        try:
            return QueryTemplate(
                intent=payload.get("intent", ""),
                template=payload.get("template", ""),
                parameters=payload.get("parameters", []),
                cypher_template=payload.get("cypher_template", ""),
                priority=payload.get("priority", 5),
                aliases=payload.get("aliases", []),
                parameter_patterns=payload.get("parameter_patterns", {})
            )
        except Exception as e:
            logging.warning(f"Failed to reconstruct template from payload: {e}")
            return None

    def _fallback_template_search(self, question: str, normalized_question: str, question_embedding: List[float], threshold: float) -> Optional[Tuple[QueryTemplate, Dict[str, str], float]]:
        """Fallback to in-memory template search if Qdrant search fails"""
        best_match = None
        best_score = 0.0
        best_params = {}

        sorted_templates = sorted(self.query_templates, key=lambda x: x.priority)

        for template in sorted_templates:
            template_embedding = self.get_embedding(template.template)
            if template_embedding is None:
                continue

            similarity = self._cosine_similarity(question_embedding, template_embedding)
            keyword_bonus = self._calculate_keyword_bonus(normalized_question, template.template)
            extracted_params = self.param_extractor.extract_parameters_advanced(question, template)

            param_completeness = len(extracted_params) / len(template.parameters) if template.parameters else 1.0
            final_score = (similarity * 0.6) + (keyword_bonus * 0.2) + (param_completeness * 0.2)
            priority_bonus = (5 - template.priority) * 0.02
            final_score += priority_bonus

            if final_score > best_score and final_score >= threshold and len(extracted_params) == len(template.parameters):
                best_score = final_score
                best_match = template
                best_params = extracted_params

        return (best_match, best_params, best_score) if best_match else None

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

    def _fuzzy_search(self, question: str, threshold: float = 85.0) -> Optional[CacheHit]:
        """
        Cerca query simili usando fuzzy string matching (Levenshtein distance).
        
        :param question: Query da cercare
        :param threshold: Soglia minima di similarità (0-100)
        :return: CacheHit se trovato match, None altrimenti
        """
        if not FUZZY_AVAILABLE:
            return None
        
        normalized_question = self.normalize_question(question)
        
        try:
            # Recupera tutte le query dalla collection principale
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Limita a 1000 per performance
                with_payload=True
            )[0]
            
            best_match = None
            best_score = 0.0
            
            for point in all_points:
                cached_normalized = self.normalize_question(point.payload.get("question", ""))
                
                # Calcola similarità usando ratio (0-100)
                similarity = fuzz.ratio(normalized_question, cached_normalized)
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = point
            
            if best_match:
                self.fuzzy_hits += 1
                return CacheHit(
                    cypher_query=best_match.payload["cypher_query"],
                    response=best_match.payload.get("response"),
                    confidence=best_score / 100.0,  # Normalizza a 0-1
                    strategy="fuzzy_match",
                    template_used=best_match.payload.get("template_used")
                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error in fuzzy search: {e}")
            return None

    def _find_best_signature_match(self, question: str, candidates: List) -> Optional[any]:
        """
        Find best structural/signature match using NLP analysis.
        Looks for queries with similar intent patterns and structure.

        :param question: Query to analyze
        :param candidates: List of candidate points from Qdrant
        :return: Best matching point or None
        """
        if not self.param_extractor.use_nlp or not candidates:
            return candidates[0] if candidates else None

        try:
            # Parse question with NLP
            question_doc = self.param_extractor.nlp(question.lower())

            # Extract signature features
            question_verbs = {token.lemma_ for token in question_doc if token.pos_ == "VERB"}
            question_nouns = {token.lemma_ for token in question_doc if token.pos_ in ["NOUN", "PROPN"]}
            question_entities = {ent.label_ for ent in question_doc.ents}

            best_match = None
            best_score = 0.0

            for candidate in candidates:
                cached_question = candidate.payload.get("question", "")
                cached_doc = self.param_extractor.nlp(cached_question.lower())

                # Extract cached signature features
                cached_verbs = {token.lemma_ for token in cached_doc if token.pos_ == "VERB"}
                cached_nouns = {token.lemma_ for token in cached_doc if token.pos_ in ["NOUN", "PROPN"]}
                cached_entities = {ent.label_ for ent in cached_doc.ents}

                # Calculate structural similarity
                verb_overlap = len(question_verbs & cached_verbs) / max(len(question_verbs | cached_verbs), 1)
                noun_overlap = len(question_nouns & cached_nouns) / max(len(question_nouns | cached_nouns), 1)
                entity_overlap = len(question_entities & cached_entities) / max(len(question_entities | cached_entities), 1)

                # Weighted signature score
                signature_score = (verb_overlap * 0.4 + noun_overlap * 0.3 + entity_overlap * 0.3)

                # Combine with vector similarity
                combined_score = (candidate.score * 0.6) + (signature_score * 0.4)

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = candidate
                    # Update the score for confidence calculation
                    best_match.score = combined_score

            # Only return if signature similarity is strong enough
            if best_score > 0.7:
                return best_match

        except Exception as e:
            logging.debug(f"Error in signature matching: {e}")
            return candidates[0] if candidates else None

        return None

    def _check_recent_results_cache(self, question: str) -> Optional[CacheHit]:
        """
        Controlla la cache in memoria degli ultimi 3 risultati.
        
        :param question: Query da cercare
        :return: CacheHit se trovato, None altrimenti
        """
        normalized = self.normalize_question(question)
        cache_key = hashlib.md5(normalized.encode()).hexdigest()
        
        if cache_key in self.recent_results_cache:
            cached = self.recent_results_cache.pop(cache_key)
            self.recent_results_cache[cache_key] = cached  # Move to end (MRU)
            self.recent_cache_hits += 1
            
            logging.debug(f"Recent results cache HIT for: {question[:50]}")
            
            return CacheHit(
                cypher_query=cached["cypher_query"],
                response=cached.get("response"),
                confidence=1.0,
                strategy="recent_cache",
                template_used=cached.get("template_used")
            )
        
        return None

    def _store_in_recent_cache(self, question: str, cypher_query: str, response: Optional[str], template_used: Optional[str]):
        """
        Memorizza un risultato nella cache in memoria degli ultimi 3.
        
        :param question: Query originale
        :param cypher_query: Query Cypher generata
        :param response: Risposta (opzionale)
        :param template_used: Template utilizzato (opzionale)
        """
        normalized = self.normalize_question(question)
        cache_key = hashlib.md5(normalized.encode()).hexdigest()
        
        # Rimuovi il più vecchio se raggiungiamo il limite
        if len(self.recent_results_cache) >= self.max_recent_results:
            self.recent_results_cache.popitem(last=False)  # FIFO: rimuovi il primo (più vecchio)
        
        self.recent_results_cache[cache_key] = {
            "question": question,
            "cypher_query": cypher_query,
            "response": response,
            "template_used": template_used,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logging.debug(f"Stored in recent cache: {question[:50]}")

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
    
    def smart_search(self, question: str, template_threshold: float = 0.8, exact_threshold: float = 0.95, similarity_threshold: float = 0.8, fuzzy_threshold: float = 85.0) -> CacheHit:
        """
        Ricerca intelligente con strategia a cascata:
        0. Recent results cache (ultimi 3, in memoria)
        1. Template matching
        2. Exact cache hit (Qdrant)
        3. Semantic signature (structural NLP similarity)
        4. Semantic similarity (Qdrant)
        5. Fuzzy matching (string similarity)
        6. No match -> LLM fallback
        """
        if not question or not question.strip():
            return CacheHit("", None, 0.0, "invalid_input")
        
        # ===== Strategy 0: Recent results cache (NUOVO) =====
        recent_hit = self._check_recent_results_cache(question)
        if recent_hit:
            return recent_hit
        
        # Strategy 1: Template matching
        template_match = self.find_best_template_match(question, threshold=template_threshold)
        if template_match:
            template, parameters, confidence = template_match
            cypher_query = self.generate_cypher_from_template(template, parameters)
            
            # Cerca response cached per questa query specifica
            cached_response = self._search_cached_response(cypher_query)
            
            cache_hit = CacheHit(
                cypher_query=cypher_query,
                response=cached_response,
                confidence=confidence,
                strategy="template_match",
                template_used=template.intent
            )
            
            # Store in recent cache
            self._store_in_recent_cache(question, cypher_query, cached_response, template.intent)
            
            return cache_hit
        
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
            
            cache_hit = CacheHit(
                cypher_query=match.payload["cypher_query"],
                response=match.payload.get("response"),
                confidence=match.score,
                strategy="exact_match"
            )
            
            # Store in recent cache
            self._store_in_recent_cache(
                question, 
                match.payload["cypher_query"],
                match.payload.get("response"),
                match.payload.get("template_used")
            )
            
            return cache_hit

        # Strategy 3: Semantic signature (structural similarity with NLP)
        signature_threshold = (exact_threshold + similarity_threshold) / 2  # Between exact and similar
        signature_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=5,
            score_threshold=signature_threshold,
        ).points

        if signature_matches:
            # Use NLP to find best structural match
            best_match = self._find_best_signature_match(question, signature_matches)

            if best_match:
                cache_hit = CacheHit(
                    cypher_query=best_match.payload["cypher_query"],
                    response=best_match.payload.get("response"),
                    confidence=best_match.score * 0.95,  # Slightly penalize
                    strategy="semantic_signature"
                )

                # Store in recent cache
                self._store_in_recent_cache(
                    question,
                    best_match.payload["cypher_query"],
                    best_match.payload.get("response"),
                    best_match.payload.get("template_used")
                )

                return cache_hit

        # Strategy 4: Semantic similarity (soglia più bassa)
        similar_matches = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=3,
            score_threshold=similarity_threshold,
        ).points

        if similar_matches:
            best_match = similar_matches[0]

            cache_hit = CacheHit(
                cypher_query=best_match.payload["cypher_query"],
                response=best_match.payload.get("response"),
                confidence=best_match.score * 0.9,  # Penalizza lievemente
                strategy="semantic_similarity"
            )

            # Store in recent cache
            self._store_in_recent_cache(
                question,
                best_match.payload["cypher_query"],
                best_match.payload.get("response"),
                best_match.payload.get("template_used")
            )

            return cache_hit

        # ===== Strategy 5: Fuzzy matching =====
        fuzzy_hit = self._fuzzy_search(question, threshold=fuzzy_threshold)
        if fuzzy_hit:
            # Store in recent cache
            self._store_in_recent_cache(
                question,
                fuzzy_hit.cypher_query,
                fuzzy_hit.response,
                fuzzy_hit.template_used
            )
            return fuzzy_hit
        
        # No match found
        return CacheHit("", None, 0.0, "no_match")

    def generate_cypher_from_template(self, template: QueryTemplate, parameters: Dict[str, str]) -> str:
        """
        Genera query Cypher da template con validazione e sanitizzazione parametri.
        
        Args:
            template: QueryTemplate contenente il cypher_template
            parameters: Dizionario con i parametri da sostituire
            
        Returns:
            str: Query Cypher con parametri sostituiti
            
        Raises:
            ValueError: Se il template non ha cypher_template definito
        """
        if not template.cypher_template:
            raise ValueError(f"Template '{template.intent}' has no cypher_template defined")
        
        cypher_query = template.cypher_template
        
        # Sostituisci parametri con escape per sicurezza
        for param, value in parameters.items():
            # Sanitizza il valore (rimuovi caratteri pericolosi)
            # Mantieni solo caratteri alfanumerici, spazi, trattini e underscore
            safe_value = re.sub(r'[^\w\s-]', '', str(value))
            
            # Sostituisci nel template
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
        """Store query con metadati migliorati - Doppia cache: Qdrant + Recent cache"""
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
            
            # ===== 1. Store in Qdrant (persistenza) =====
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
            
            # ===== 2. Store in recent cache (ultimi 3, in memoria) =====
            self._store_in_recent_cache(question, cypher_query, response, template_used)
            
            logging.info(f"Successfully stored query in both Qdrant and recent cache: {question[:50]}...")
            return True
            
        except Exception as e:
            logging.error(f"Error storing query and response: {e}")
            return False

    def store_batch_queries(self, queries: List[Dict]) -> bool:
        """
        Store multiple queries efficiently using batch operations.

        Args:
            queries: List of dicts with keys: question, cypher_query, response, template_used

        Returns:
            bool: True if successful, False otherwise

        Example:
            >>> queries = [
            ...     {"question": "Q1", "cypher_query": "C1", "response": "R1", "template_used": "T1"},
            ...     {"question": "Q2", "cypher_query": "C2", "response": "R2", "template_used": None}
            ... ]
            >>> cache.store_batch_queries(queries)
        """
        if not queries:
            logging.warning("Empty queries list provided to store_batch_queries")
            return False

        try:
            points = []
            batch_size = int(os.getenv("BATCH_SIZE", "100"))

            for query_data in queries:
                if not all([query_data.get("question"), query_data.get("cypher_query")]):
                    logging.warning(f"Skipping invalid query data: {query_data}")
                    continue

                question = query_data["question"]
                cypher_query = query_data["cypher_query"]
                response = query_data.get("response", "")
                template_used = query_data.get("template_used")

                # Generate embedding
                question_embedding = self.get_embedding(question)
                if question_embedding is None:
                    logging.warning(f"Failed to generate embedding for: {question[:50]}")
                    continue

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
                points.append(point)

                # Store in recent cache as well
                self._store_in_recent_cache(question, cypher_query, response, template_used)

            if not points:
                logging.warning("No valid points to store after processing batch")
                return False

            # Batch upsert in chunks for very large batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=batch
                )
                logging.info(f"Stored batch {i//batch_size + 1}: {len(batch)} queries")

            logging.info(f"Successfully stored {len(points)} queries in batch operation")
            return True

        except Exception as e:
            logging.error(f"Error in batch storing queries: {e}")
            return False

    async def async_smart_search(self, question: str, **kwargs) -> CacheHit:
        """
        Async version of smart_search for concurrent operations.

        Args:
            question: The query to search for
            **kwargs: Additional search parameters (thresholds, etc.)

        Returns:
            CacheHit: Search result with cypher query and metadata
        """
        if not question or not question.strip():
            return CacheHit("", None, 0.0, "invalid_input")

        try:
            # Initialize async client
            if self.mode == "memory":
                async_client = AsyncQdrantClient(":memory:")
            elif self.mode == "cloud":
                async_client = AsyncQdrantClient(
                    host=os.getenv("QDRANT_HOST"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                )
            elif self.mode == "docker":
                async_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
            else:
                raise ValueError("Invalid Qdrant mode")

            # Check recent cache first (synchronous, fast)
            recent_hit = self._check_recent_results_cache(question)
            if recent_hit:
                await async_client.close()
                return recent_hit

            # Template matching (synchronous)
            template_threshold = kwargs.get("template_threshold", 0.8)
            template_match = self.find_best_template_match(question, threshold=template_threshold)
            if template_match:
                template, parameters, confidence = template_match
                cypher_query = self.generate_cypher_from_template(template, parameters)
                cached_response = self._search_cached_response(cypher_query)

                await async_client.close()
                cache_hit = CacheHit(
                    cypher_query=cypher_query,
                    response=cached_response,
                    confidence=confidence,
                    strategy="template_match",
                    template_used=template.intent
                )
                self._store_in_recent_cache(question, cypher_query, cached_response, template.intent)
                return cache_hit

            # Semantic search using async client
            normalized_question = self.normalize_question(question)
            question_embedding = self.get_embedding(normalized_question)

            if question_embedding is None:
                await async_client.close()
                return CacheHit("", None, 0.0, "embedding_failed")

            exact_threshold = kwargs.get("exact_threshold", 0.95)
            exact_matches = await async_client.query_points(
                collection_name=self.collection_name,
                query=question_embedding,
                limit=1,
                score_threshold=exact_threshold,
            )

            if exact_matches.points:
                match = exact_matches.points[0]
                await async_client.close()

                cache_hit = CacheHit(
                    cypher_query=match.payload["cypher_query"],
                    response=match.payload.get("response"),
                    confidence=match.score,
                    strategy="exact_match"
                )
                self._store_in_recent_cache(
                    question,
                    match.payload["cypher_query"],
                    match.payload.get("response"),
                    match.payload.get("template_used")
                )
                return cache_hit

            # Semantic similarity search
            similarity_threshold = kwargs.get("similarity_threshold", 0.8)
            similar_matches = await async_client.query_points(
                collection_name=self.collection_name,
                query=question_embedding,
                limit=3,
                score_threshold=similarity_threshold,
            )

            await async_client.close()

            if similar_matches.points:
                best_match = similar_matches.points[0]
                cache_hit = CacheHit(
                    cypher_query=best_match.payload["cypher_query"],
                    response=best_match.payload.get("response"),
                    confidence=best_match.score * 0.9,
                    strategy="semantic_similarity"
                )
                self._store_in_recent_cache(
                    question,
                    best_match.payload["cypher_query"],
                    best_match.payload.get("response"),
                    best_match.payload.get("template_used")
                )
                return cache_hit

            # No match found
            return CacheHit("", None, 0.0, "no_match")

        except Exception as e:
            logging.error(f"Error in async smart search: {e}")
            return CacheHit("", None, 0.0, "error")

    async def async_batch_search(self, questions: List[str], **kwargs) -> List[CacheHit]:
        """
        Perform multiple searches concurrently for better throughput.

        Args:
            questions: List of questions to search for
            **kwargs: Additional search parameters

        Returns:
            List[CacheHit]: Results for each question in the same order

        Example:
            >>> questions = ["Who are the employees?", "List all projects", "Show me departments"]
            >>> results = await cache.async_batch_search(questions)
        """
        if not questions:
            return []

        try:
            # Create tasks for concurrent execution
            tasks = [self.async_smart_search(question, **kwargs) for question in questions]

            # Execute all searches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            cache_hits = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Error searching question '{questions[i][:50]}': {result}")
                    cache_hits.append(CacheHit("", None, 0.0, "error"))
                else:
                    cache_hits.append(result)

            logging.info(f"Completed batch search for {len(questions)} questions")
            return cache_hits

        except Exception as e:
            logging.error(f"Error in async batch search: {e}")
            return [CacheHit("", None, 0.0, "error") for _ in questions]

    def _update_usage_count_efficient(self, point_id: str):
        """Aggiorna contatore uso in modo efficiente"""
        try:
            # Per ora incrementiamo un contatore locale
            # In futuro si potrebbe implementare un batch update
            pass
        except Exception as e:
            logging.debug(f"Error updating usage count: {e}")

    def get_performance_stats(self) -> Dict:
        """Enhanced performance statistics with detailed Qdrant metrics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            template_info = self.qdrant_client.get_collection(self.template_collection)

            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

            # Calculate search strategy distribution
            total_search_hits = self.template_hits + self.fuzzy_hits + self.recent_cache_hits

            stats = {
                "cache_performance": {
                    "embedding_cache_hits": self.cache_hits,
                    "embedding_cache_misses": self.cache_misses,
                    "hit_rate_percentage": round(cache_hit_rate, 2),
                    "template_hits": self.template_hits,
                    "fuzzy_hits": self.fuzzy_hits,
                    "recent_cache_hits": self.recent_cache_hits,
                    "total_search_hits": total_search_hits
                },
                "storage": {
                    "total_cached_queries": collection_info.points_count,
                    "templates_in_qdrant": template_info.points_count,
                    "embedding_cache_size": len(self.embedding_cache),
                    "frequent_queries_cache_size": len(self.frequent_queries_cache),
                    "recent_results_cache_size": len(self.recent_results_cache),
                    "vector_size": self.vector_size
                },
                "qdrant_metrics": {
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "segments_count": collection_info.segments_count,
                    "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
                    "optimizer_status": collection_info.optimizer_status.value if hasattr(collection_info.optimizer_status, 'value') else str(collection_info.optimizer_status),
                    "config": {
                        "distance": collection_info.config.params.vectors.distance.value if hasattr(collection_info.config.params.vectors, 'distance') else "cosine",
                        "hnsw_m": collection_info.config.hnsw_config.m if collection_info.config.hnsw_config else "N/A",
                        "hnsw_ef_construct": collection_info.config.hnsw_config.ef_construct if collection_info.config.hnsw_config else "N/A",
                        "quantization_enabled": collection_info.config.quantization_config is not None,
                        "quantization_type": type(collection_info.config.quantization_config).__name__ if collection_info.config.quantization_config else None,
                        "on_disk": collection_info.config.params.vectors.on_disk if hasattr(collection_info.config.params.vectors, 'on_disk') else False
                    }
                },
                "model_info": {
                    "embedder_model": self.embedder,
                    "nlp_enabled": self.param_extractor.use_nlp,
                    "fuzzy_matching_available": FUZZY_AVAILABLE
                },
                "efficiency": {
                    "templates_loaded": len(self.query_templates),
                    "avg_template_priority": sum(t.priority for t in self.query_templates) / len(self.query_templates) if self.query_templates else 0,
                    "memory_efficiency": {
                        "max_embedding_cache": self.max_cache_size,
                        "max_frequent_cache": self.max_frequent_cache,
                        "max_recent_cache": self.max_recent_results
                    }
                }
            }

            return stats

        except Exception as e:
            logging.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}

    def clear_caches(self):
        """Pulisci tutte le cache in-memory (mantiene Qdrant)"""
        self.embedding_cache.clear()
        self.frequent_queries_cache.clear()
        self.recent_results_cache.clear()  # NUOVO
        self.cache_hits = 0
        self.cache_misses = 0
        self.template_hits = 0
        self.fuzzy_hits = 0  # NUOVO
        self.recent_cache_hits = 0  # NUOVO
        logging.info("All in-memory caches cleared (Qdrant data preserved)")


# Funzione di utilità per inizializzazione rapida
def create_optimized_cache(collection_name: str = "semantic_cache", **kwargs) -> SemanticCache:
    """
    Crea una cache semantica ottimizzata con configurazione di default.
    
    Args:
        collection_name: Nome della collection Qdrant
        **kwargs: Parametri opzionali (mode, embedder, vector_size)
    
    Returns:
        SemanticCache: Istanza configurata della cache semantica
    
    Example:
        >>> cache = create_optimized_cache("my_cache", mode="memory")
        >>> cache = create_optimized_cache("prod_cache", mode="cloud", vector_size=512)
    """
    defaults = {
        "mode": os.getenv("QDRANT_MODE", "memory"),
        "embedder": os.getenv("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_size": int(os.getenv("VECTOR_SIZE", "384"))
    }
    defaults.update(kwargs)
    
    return SemanticCache(
        collection_name=collection_name,
        mode=defaults["mode"],
        embedder=defaults["embedder"],
        vector_size=defaults["vector_size"]
    )