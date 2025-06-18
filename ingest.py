# ingest.py

import os
import json
import shutil
import re
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

# --- LlamaIndex Imports ---
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# --- Pydantic Models ---
from pydantic import BaseModel, Field, ValidationError

class LifeStorySchema(BaseModel):
    birth_date: Optional[str] = Field(None, description="Birth date in YYYY-MM-DD format")
    birth_place: Optional[str] = Field(None)
    death_date: Optional[str] = Field(None, description="Death date in YYYY-MM-DD format")
    creation_process: Optional[str] = Field(None)
    parents_donors: List[str] = Field(default_factory=list)
    children_names: List[str] = Field(default_factory=list)
    legacy: Optional[str] = Field(None)

class TravelPhase(BaseModel):
    phase_name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    lamb_activities: List[str] = Field(default_factory=list)

class SunsetPhase(BaseModel):
    description: Optional[str] = Field(None)
    dollys_feeling: Optional[str] = Field(None)

class ReturnPhase(BaseModel):
    was_it_worth_it: Optional[bool] = Field(None)
    lasting_memory: Optional[str] = Field(None)

class VacationStorySchema(BaseModel):
    destination: Optional[str] = Field(None)
    attendees: List[str] = Field(default_factory=list)
    travel_phase: Optional[TravelPhase] = Field(None)
    landing_phase: Optional[str] = Field(None)
    beach_phase: Optional[TravelPhase] = Field(None)
    sunset_phase: Optional[SunsetPhase] = Field(None)
    return_phase: Optional[ReturnPhase] = Field(None)

class InterviewSchema(BaseModel):
    interviewer: Optional[str] = Field(None)
    topic: Optional[str] = Field(None)
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    how_she_started: Optional[str] = Field(None)
    creations_mentioned: List[str] = Field(default_factory=list)
    future_ambition: Optional[str] = Field(None)
    artistic_philosophy: Optional[str] = Field(None)

class JournalEntry(BaseModel):
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    summary: Optional[str] = Field(None)

class JournalSchema(BaseModel):
    location: Optional[str] = Field(None)
    partner: Optional[str] = Field(None)
    journal_entries: List[JournalEntry] = Field(default_factory=list)

class RelationshipOriginSchema(BaseModel):
    partner: Optional[str] = Field(None)
    how_they_met: Optional[str] = Field(None)
    defining_moment: Optional[str] = Field(None)
    dollys_view_of_partner: Optional[str] = Field(None)

class GenealogySchema(BaseModel):
    spouse: Optional[str] = Field(None)
    children: List[str] = Field(default_factory=list)
    parents: List[str] = Field(default_factory=list)
    grandparents: List[str] = Field(default_factory=list)
    grandchildren: List[str] = Field(default_factory=list)
    special_notes: List[str] = Field(default_factory=list)
    personalized_message: Optional[str] = Field(None)

class FactualDocumentSchema(BaseModel):
    key_facts: List[str] = Field(default_factory=list)
    people_mentioned: List[str] = Field(default_factory=list)
    organizations_mentioned: List[str] = Field(default_factory=list)

# --- Configuration ---
class Config:
    DATA_DIR = Path("data")
    EXTRACTED_FACTS_DIR = Path("data/extracted_facts")
    LLM_RAW_OUTPUT_DIR = Path("data/llm_raw_output")
    FAISS_DB_DIR = Path("faiss_db")
    OUTPUT_FILE = Path("dolly_facts.json")
    
    # LLM Configuration
    LLM_MODEL = "llama3.1:8b"
    LLM_TIMEOUT = 1200.0
    EMBEDDING_MODEL = "nomic-embed-text-v1.5"
    EMBEDDING_DIMENSION = 768
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2

# --- Setup Logging ---
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ingest.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Enhanced JSON Extraction ---
class JSONExtractor:
    """Robust JSON extraction from LLM responses"""
    
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Clean common JSON formatting issues"""
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix common quote issues
        json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # Unquoted keys
        
        # Remove any text before first { or after last }
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]
        
        return json_str.strip()
    
    @staticmethod
    def extract_json_from_response(llm_output: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response with multiple strategies"""
        if not llm_output or not llm_output.strip():
            logger.warning("Empty LLM output received")
            return None
        
        strategies = [
            JSONExtractor._extract_full_json,
            JSONExtractor._extract_with_regex,
            JSONExtractor._extract_partial_json,
            JSONExtractor._create_fallback_json
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                result = strategy(llm_output)
                if result:
                    logger.info(f"JSON extraction successful using strategy {i}")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {i} failed: {e}")
                continue
        
        logger.error("All JSON extraction strategies failed")
        return None
    
    @staticmethod
    def _extract_full_json(llm_output: str) -> Optional[Dict[str, Any]]:
        """Try to extract complete JSON object"""
        # Find JSON block
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            return None
        
        json_str = JSONExtractor.clean_json_string(json_match.group(0))
        return json.loads(json_str)
    
    @staticmethod
    def _extract_with_regex(llm_output: str) -> Optional[Dict[str, Any]]:
        """Extract JSON using more flexible regex patterns"""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json_str = JSONExtractor.clean_json_string(match)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        return None
    
    @staticmethod
    def _extract_partial_json(llm_output: str) -> Optional[Dict[str, Any]]:
        """Try to construct JSON from key-value pairs in text"""
        # Look for field patterns
        field_patterns = [
            r'"?(\w+)"?\s*:\s*"([^"]*)"',
            r'"?(\w+)"?\s*:\s*\[([^\]]*)\]',
            r'"?(\w+)"?\s*:\s*(true|false|null|\d+)',
        ]
        
        extracted_data = {}
        for pattern in field_patterns:
            matches = re.findall(pattern, llm_output, re.IGNORECASE)
            for key, value in matches:
                try:
                    # Try to parse the value
                    if value.lower() in ['true', 'false']:
                        extracted_data[key] = value.lower() == 'true'
                    elif value.lower() == 'null':
                        extracted_data[key] = None
                    elif value.isdigit():
                        extracted_data[key] = int(value)
                    elif value.startswith('[') and value.endswith(']'):
                        # Simple list parsing
                        items = [item.strip(' "') for item in value[1:-1].split(',')]
                        extracted_data[key] = [item for item in items if item]
                    else:
                        extracted_data[key] = value
                except Exception:
                    extracted_data[key] = value
        
        return extracted_data if extracted_data else None
    
    @staticmethod
    def _create_fallback_json(llm_output: str) -> Dict[str, Any]:
        """Create a minimal fallback JSON structure"""
        return {
            "extraction_failed": True,
            "raw_output": llm_output[:500] + ("..." if len(llm_output) > 500 else ""),
            "error_message": "Failed to parse structured data from LLM output"
        }

# --- Document Processing ---
class DocumentProcessor:
    """Handle document loading and processing"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.pdf_loader = PyMuPDFReader()
    
    def load_documents(self) -> Dict[str, str]:
        """Load all documents from data directory"""
        docs_by_filename = {}
        
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            return docs_by_filename
        
        # Load text files
        for file_path in self.data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        docs_by_filename[file_path.name] = content
                        logger.info(f"Loaded text file: {file_path.name}")
                    else:
                        logger.warning(f"Empty text file: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading text file {file_path.name}: {e}")
        
        # Load PDF files
        for file_path in self.data_dir.glob("*.pdf"):
            try:
                pdf_docs = self.pdf_loader.load(file_path=str(file_path))
                content = "\n".join([doc.get_content() for doc in pdf_docs]).strip()
                if content:
                    docs_by_filename[file_path.name] = content
                    logger.info(f"Loaded PDF file: {file_path.name}")
                else:
                    logger.warning(f"Empty PDF file: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading PDF file {file_path.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(docs_by_filename)}")
        return docs_by_filename

# --- Data Extraction ---
class DataExtractor:
    """Handle structured data extraction from documents"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.schema_map = {
            "Baa_I_Am_Dolly.txt": (LifeStorySchema, "Extract key life story facts from Dolly's memoir, including birth/death dates, children, and legacy."),
            "Dolly_Hawaiian_Baacation.txt": (VacationStorySchema, "Extract details about the Hawaiian vacation, following the multi-phase schema provided."),
            "Dolly_Pottery_Interview.txt": (InterviewSchema, "Extract key points from the interview with Fiona McGuffin."),
            "Honey_Moon_Journal.txt": (JournalSchema, "Summarize the key events from each dated entry in the honeymoon journal."),
            "wedding_vows_to_my_ray_ray.txt": (RelationshipOriginSchema, "Extract the story of Dolly's relationship with Raymond from her wedding vows."),
            "Dollly_Geneology.pdf": (GenealogySchema, "Extract all family relationships from this genealogy report. Pay close attention to headings like 'Parents', 'Grandparents', 'Offspring' to correctly assign the names."),
            "Dollly_Thesis.pdf": (FactualDocumentSchema, "Extract factual data points and the thesis title from this academic paper."),
            "The_New_Porker_Article.pdf": (FactualDocumentSchema, "Extract factual data points about the protest from this news article."),
            "Dolly- The science behind the world's most famous sheep.pdf": (FactualDocumentSchema, "Extract key scientific facts and dates from this paper."),
        }
        
        self.base_prompt = """You are a professional data extraction expert. Your task is to analyze the provided document and extract structured information according to the specified schema.

CRITICAL REQUIREMENTS:
1. Output ONLY a valid JSON object that strictly follows the provided schema
2. Do NOT include any explanatory text, markdown formatting, or code blocks
3. Convert all dates to YYYY-MM-DD format (e.g., "July 5, 1996" → "1996-07-05")
4. Extract ONLY information explicitly stated in the document
5. Use null for missing optional fields, empty arrays for missing lists
6. Ensure all JSON syntax is correct (proper quotes, commas, brackets)

EXTRACTION TASK: {instruction}

JSON SCHEMA:
{schema}

DOCUMENT CONTENT:
{content}

JSON OUTPUT:"""
    
    def extract_from_document(self, filename: str, content: str, 
                            output_dir: Path, raw_output_dir: Path) -> bool:
        """Extract structured data from a single document"""
        if not content.strip():
            logger.warning(f"Skipping empty document: {filename}")
            return False
        
        schema_class, instruction = self.schema_map.get(
            filename, (FactualDocumentSchema, "Extract general factual information.")
        )
        
        logger.info(f"Processing {filename} with {schema_class.__name__}")
        
        # Create prompt
        prompt = self.base_prompt.format(
            instruction=instruction,
            schema=json.dumps(schema_class.model_json_schema(), indent=2),
            content=content[:8000]  # Limit content length to avoid token limits
        )
        
        # Extract with retries
        for attempt in range(Config.MAX_RETRIES):
            try:
                logger.debug(f"Attempt {attempt + 1} for {filename}")
                
                # Get LLM response
                response = self.llm.complete(prompt)
                response_str = str(response).strip()
                
                # Save raw output
                raw_output_file = raw_output_dir / f"{filename}.txt"
                with open(raw_output_file, "w", encoding='utf-8') as f:
                    f.write(response_str)
                
                # Extract JSON
                json_data = JSONExtractor.extract_json_from_response(response_str)
                if not json_data:
                    raise ValueError("Failed to extract valid JSON")
                
                # Validate against schema
                try:
                    validated_data = schema_class.model_validate(json_data)
                    
                    # Save validated data
                    output_file = output_dir / f"{filename}.json"
                    with open(output_file, "w", encoding='utf-8') as f:
                        json.dump(
                            validated_data.model_dump(exclude_none=True, exclude_unset=True), 
                            f, indent=2, ensure_ascii=False
                        )
                    
                    logger.info(f"Successfully extracted data from {filename}")
                    return True
                    
                except ValidationError as e:
                    logger.warning(f"Validation failed for {filename}: {e}")
                    # Save partial data even if validation fails
                    output_file = output_dir / f"{filename}.json"
                    with open(output_file, "w", encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {filename}: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    import time
                    time.sleep(Config.RETRY_DELAY)
                continue
        
        logger.error(f"All extraction attempts failed for {filename}")
        return False

# --- Knowledge Graph Merger ---
class KnowledgeGraphMerger:
    """Merge extracted data into a unified knowledge graph"""
    
    @staticmethod
    def merge_knowledge_graphs(extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple data dictionaries into a master knowledge graph"""
        if not extracted_data:
            return {}
        
        master_graph = {}
        
        for data in extracted_data:
            if not isinstance(data, dict):
                continue
                
            for key, value in data.items():
                if key in master_graph:
                    master_graph[key] = KnowledgeGraphMerger._merge_values(
                        master_graph[key], value
                    )
                else:
                    master_graph[key] = value
        
        # Post-process to remove duplicates and clean data
        return KnowledgeGraphMerger._clean_merged_data(master_graph)
    
    @staticmethod
    def _merge_values(existing: Any, new: Any) -> Any:
        """Merge two values based on their types"""
        # Both are lists - merge them
        if isinstance(existing, list) and isinstance(new, list):
            combined = existing + new
            return KnowledgeGraphMerger._deduplicate_list(combined)
        
        # Both are dicts - merge recursively
        elif isinstance(existing, dict) and isinstance(new, dict):
            merged = existing.copy()
            for k, v in new.items():
                if k in merged:
                    merged[k] = KnowledgeGraphMerger._merge_values(merged[k], v)
                else:
                    merged[k] = v
            return merged
        
        # Different types or one is None - prefer non-None and more recent
        elif existing is None:
            return new
        elif new is None:
            return existing
        else:
            # For conflicting non-None values, prefer the new one
            return new
    
    @staticmethod
    def _deduplicate_list(items: List[Any]) -> List[Any]:
        """Remove duplicates from list while preserving order"""
        if not items:
            return []
        
        # For list of dicts, deduplicate by content
        if items and isinstance(items[0], dict):
            seen = set()
            unique_items = []
            for item in items:
                # Create a hashable representation
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
            return unique_items
        
        # For simple items, use order-preserving deduplication
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items
    
    @staticmethod
    def _clean_merged_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize merged data"""
        cleaned = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                # Remove empty strings and None values
                cleaned_list = [item for item in value if item not in [None, "", " "]]
                if cleaned_list:
                    cleaned[key] = cleaned_list
            elif isinstance(value, dict):
                cleaned_dict = KnowledgeGraphMerger._clean_merged_data(value)
                if cleaned_dict:
                    cleaned[key] = cleaned_dict
            elif value not in [None, "", " "]:
                cleaned[key] = value
        
        return cleaned

# --- Vector Store Builder ---
class VectorStoreBuilder:
    """Build FAISS vector store from documents"""
    
    def __init__(self, embedding_model, faiss_db_dir: Path):
        self.embedding_model = embedding_model
        self.faiss_db_dir = faiss_db_dir
    
    def build_vector_store(self, docs_by_filename: Dict[str, str]):
        """Build FAISS vector store from document contents"""
        if not docs_by_filename:
            logger.warning("No documents provided for vector store creation")
            return
        
        try:
            # Create LlamaIndex documents
            llama_documents = [
                Document(text=content, metadata={"filename": filename})
                for filename, content in docs_by_filename.items()
                if content.strip()
            ]
            
            if not llama_documents:
                logger.warning("No valid documents for vector store creation")
                return
            
            logger.info(f"Creating vector store from {len(llama_documents)} documents")
            
            # Create FAISS index
            faiss_index = faiss.IndexFlatL2(Config.EMBEDDING_DIMENSION)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build index
            narrative_index = VectorStoreIndex.from_documents(
                llama_documents, 
                storage_context=storage_context
            )
            
            # Persist to disk
            narrative_index.storage_context.persist(persist_dir=str(self.faiss_db_dir))
            
            logger.info(f"FAISS vector store created with {faiss_index.ntotal} embedded chunks")
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise

# --- Main Pipeline ---
class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.setup_environment()
        self.setup_models()
    
    def setup_environment(self):
        """Setup directories and environment"""
        logger.info("Setting up environment...")
        
        # Clean and create directories
        directories = [
            self.config.EXTRACTED_FACTS_DIR,
            self.config.LLM_RAW_OUTPUT_DIR,
            self.config.FAISS_DB_DIR
        ]
        
        for directory in directories:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_models(self):
        """Initialize LLM and embedding models"""
        logger.info("Initializing models...")
        
        # Check for required API key
        nomic_api_key = os.getenv("NOMIC_API_KEY")
        if not nomic_api_key:
            raise ValueError("NOMIC_API_KEY environment variable not set")
        
        # Setup LlamaIndex settings
        Settings.llm = Ollama(
            model=self.config.LLM_MODEL, 
            request_timeout=self.config.LLM_TIMEOUT
        )
        Settings.embed_model = NomicEmbedding(
            model_name=self.config.EMBEDDING_MODEL,
            nomic_api_key=nomic_api_key
        )
        
        logger.info("Models initialized successfully")
    
    def run(self):
        """Execute the complete ingestion pipeline"""
        try:
            logger.info("Starting Dolly Facts Ingestion Pipeline")
            
            # Phase 1: Load documents
            logger.info("Phase 1: Loading documents...")
            doc_processor = DocumentProcessor(self.config.DATA_DIR)
            docs_by_filename = doc_processor.load_documents()
            
            if not docs_by_filename:
                logger.error("No documents loaded. Exiting.")
                return False
            
            # Phase 2: Extract structured data
            logger.info("Phase 2: Extracting structured data...")
            extractor = DataExtractor(Settings.llm)
            
            successful_extractions = 0
            for filename, content in docs_by_filename.items():
                if extractor.extract_from_document(
                    filename, content,
                    self.config.EXTRACTED_FACTS_DIR,
                    self.config.LLM_RAW_OUTPUT_DIR
                ):
                    successful_extractions += 1
            
            logger.info(f"Successfully extracted data from {successful_extractions}/{len(docs_by_filename)} documents")
            
            # Phase 3: Merge knowledge graphs
            logger.info("Phase 3: Merging knowledge graphs...")
            all_extracted_data = []
            
            for json_file in self.config.EXTRACTED_FACTS_DIR.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_extracted_data.append(data)
                        logger.debug(f"Loaded data from {json_file.name}")
                except Exception as e:
                    logger.warning(f"Could not load {json_file.name}: {e}")
            
            # Merge data
            master_knowledge_graph = KnowledgeGraphMerger.merge_knowledge_graphs(all_extracted_data)
            
            # Save master knowledge graph
            with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(master_knowledge_graph, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Master knowledge graph saved to {self.config.OUTPUT_FILE}")
            
            # Phase 4: Build vector store
            logger.info("Phase 4: Building vector store...")
            vector_builder = VectorStoreBuilder(Settings.embed_model, self.config.FAISS_DB_DIR)
            vector_builder.build_vector_store(docs_by_filename)
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

def main():
    """Main entry point"""
    try:
        pipeline = IngestionPipeline()
        success = pipeline.run()
        
        if success:
            print("\n✅ Ingestion pipeline completed successfully!")
            print(f"📊 Structured data: {Config.OUTPUT_FILE}")
            print(f"🔍 Vector store: {Config.FAISS_DB_DIR}")
            print(f"📝 Logs: ingest.log")
        else:
            print("\n❌ Pipeline completed with errors. Check logs for details.")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⏹️  Pipeline interrupted")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\n💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
