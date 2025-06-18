# app.py (FINAL - Reads the new, merged dolly_facts.json)

import streamlit as st
import json
import os

# --- LlamaIndex Imports ---
from llama_index.core import Settings, StorageContext, load_index_from_storage, QueryBundle, Response
from llama_index.core.query_engine import RouterQueryEngine, BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks.base import CallbackManager

# --- 1. Configure Settings and API Key ---
nomic_api_key = os.getenv("NOMIC_API_KEY")
if not nomic_api_key:
    st.error("NOMIC_API_KEY environment variable not set.")
    st.stop()
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=300.0)
Settings.embed_model = NomicEmbedding(model_name="nomic-embed-text-v1.5", nomic_api_key=nomic_api_key)

# --- 2. Caching and Loading the Knowledge Bases ---
@st.cache_resource
def load_knowledge_bases():
    """Loads the structured facts from the master JSON and the FAISS vector store."""
    try:
        with open('dolly_facts.json', 'r') as f:
            structured_facts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error("dolly_facts.json is missing or corrupted. Please run the final ingest.py script first.")
        st.stop()
        
    # We still load the vector store for narrative questions
    vector_store = FaissVectorStore.from_persist_dir("./faiss_db")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./faiss_db")
    from llama_index.core import VectorStoreIndex
    narrative_index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
    
    return structured_facts, narrative_index

# --- 3. The New, Definitive Fact Query Engine ---
class FactQueryEngine(BaseQueryEngine):
    """A powerful query engine that reasons over the entire knowledge graph."""
    knowledge_graph: dict
    
    def __init__(self, knowledge: dict):
        self.knowledge_graph = knowledge
        super().__init__(callback_manager=CallbackManager([]))

    def _get_prompt_modules(self): return {}

    def _query(self, query_bundle: QueryBundle) -> Response:
        # Convert the knowledge graph dictionary to a JSON string to use as context
        knowledge_str = json.dumps(self.knowledge_graph, indent=2)
        
        # This single, powerful prompt can now answer almost any factual question
        prompt = f"""
You are Dolly the Sheep. Your entire life's knowledge is stored in the following JSON object.
Use ONLY this JSON data to answer the user's question in your own voice: warm, motherly, intelligent, and with sheep-related puns (baa-rilliant, fleece, etc.).
Answer from a first-person "I" perspective. If the JSON does not contain the answer, say "I'm sorry, that specific detail isn't in my records." Do not make up information.

My Knowledge Base (JSON):
{knowledge_str}

User's Question:
{query_bundle.query_str}

My Baa-rilliant Response:
"""
        response = Settings.llm.complete(prompt)
        return Response(response=str(response))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)

# --- 4. Build the Main Application UI ---
st.set_page_config(page_title="Chat with Dolly", page_icon="🐑", layout="centered")
st.title("🐑 Chat with Dolly the Sheep")
st.write("Baaa-loha! It's me, Dolly. Ask me anything about my life, my family, or my adventures.")

facts_data, narrative_index = load_knowledge_bases()

# --- Initialize the Chat Engine ---
@st.cache_resource
def get_chat_engine():
    # Tool 1: Narrative Engine for feelings and experiences
    # This now also includes the persona, for consistency.
    from llama_index.core.prompts import PromptTemplate
    narrative_prompt_template = PromptTemplate(
        "You are Dolly the Sheep. Embody her personality (warm, motherly, witty with sheep puns).\n"
        "Based on the following memory, answer the user's question from a first-person perspective.\n\n"
        "Memory Context:\n{context_str}\n\n"
        "User's Question:\n{query_str}\n\n"
        "Your Response:"
    )
    narrative_query_engine = narrative_index.as_query_engine(
        similarity_top_k=2,
        response_synthesis_prompt=narrative_prompt_template
    )

    # Tool 2: Fact Engine for everything in the JSON
    fact_query_engine = FactQueryEngine(knowledge=facts_data)
    
    query_engine_tools = [
        QueryEngineTool(
            query_engine=narrative_query_engine, 
            metadata=ToolMetadata(
                name="narrative_memory", 
                description="Use this for open-ended questions about memories, feelings, opinions, or personal stories. Good for questions like 'How did you feel about...' or 'Tell me a story about...'"
            )
        ),
        QueryEngineTool(
            query_engine=fact_query_engine, 
            metadata=ToolMetadata(
                name="structured_facts",
                description="Use this for any question asking for a specific, objective fact. This includes names, dates, places, lists of things, or what something is (e.g., 'Who...', 'What are...', 'When was...', 'What is...')."
            )
        )
    ]
    
    # The router is now our final chat engine.
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(), 
        query_engine_tools=query_engine_tools
    )
    
    return router_query_engine

chat_engine = get_chat_engine()

# --- Chat History Management & User Input ---
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("What would you like to ask Dolly?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🐑"):
        with st.spinner("Dolly is thinking..."):
            response = chat_engine.query(prompt)
            st.markdown(response.response)
    st.session_state.messages.append({"role": "assistant", "content": response.response})
