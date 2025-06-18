# Dolly the Sheep - Digital Twin

This project is an interactive digital twin of Dolly the Sheep, built as a take-home assignment. The system ingests a variety of documents—including memoirs, interviews, scientific papers, and personal journals—to create a comprehensive, multi-modal knowledge base. An interactive chat application then allows users to ask Dolly questions about her life, experiences, and feelings, with the system responding in her unique persona.

The core of this project is a sophisticated agentic architecture designed to provide accurate, in-character responses while minimizing AI hallucination. It demonstrates an advanced approach to Retrieval-Augmented Generation (RAG) by separating factual knowledge from narrative context.

## Core Architecture

The system is built entirely on a local, open-source stack and avoids simple vector search in favor of a more robust, two-part knowledge base and an agentic query engine.

1.  **Dual Knowledge Base:** To ensure both factual accuracy and narrative richness, Dolly's knowledge is split into two distinct components:
    *   **Structured Knowledge Graph (`dolly_facts.json`):** A detailed JSON file containing extracted entities, relationships, and key facts (family, dates, events, etc.). This knowledge graph is created by a multi-schema ingestion pipeline that uses a Llama 3.1 model to intelligently parse each document according to its specific content type (e.g., using a `GenealogySchema` for the family tree PDF, a `VacationStorySchema` for the travelogue, etc.). This content-aware approach yields a much higher quality of structured data than a generic approach.
    *   **Unstructured Narrative Store (`faiss_db/`):** A FAISS vector database containing embeddings of the raw text from all documents. This store captures Dolly's unique voice, her feelings, and the rich, descriptive context of her stories.

2.  **Agentic RAG Chat Engine:** The chatbot is not a simple Q&A bot. It uses a `RouterQueryEngine` to provide high-quality, contextually-aware responses:
    *   A **Router** first analyzes the user's question to determine if it is factual ("Who are your children?") or narrative-based ("How did you feel on your honeymoon?").
    *   For factual questions, it queries the structured **Knowledge Graph** using a custom-built, deterministic engine that provides precise answers without LLM hallucination.
    *   For narrative questions, it performs a similarity search on the **FAISS vector store** to retrieve relevant memories.
    *   A final **Persona Layer** is applied only to the narrative responses, ensuring all answers are delivered in Dolly's unique, witty, and warm voice.

## Tech Stack

-   **LLM:** Llama 3.1 8B (running locally via `Ollama`)
-   **Framework:** LlamaIndex
-   **Vector Store:** FAISS (Facebook AI Similarity Search)
-   **Embedding Model:** Nomic Embed
-   **UI:** Streamlit
-   **Core Language:** Python 3.11

## Setup and Installation Instructions

These instructions will guide you through setting up the project on a macOS system.

### Step 1: Install System Dependencies (Homebrew & Python)

First, ensure you have the necessary system-level tools for compiling Python packages.

```bash
# Install Homebrew (macOS package manager) if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# IMPORTANT: Follow the on-screen instructions from the installer to add Homebrew to your PATH.

# Install Python 3.11
brew install python@3.11
```

### Step 2: Set Up the Project Environment

Clone this repository and set up a Python virtual environment.

```bash
# Navigate to a directory where you want to store the project
cd ~/Desktop

# Clone the repository
# git clone [URL_OF_YOUR_GITHUB_REPO]
cd dolly-digital-twin

# Create a virtual environment using the freshly installed Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 3: Install Python Dependencies

Install all required Python packages using the provided `requirements.txt` file.

```bash
# Ensure your virtual environment is active before running
pip install --upgrade pip
pip install -r requirements.txt
pip install PyMuPDF # Required for the advanced PDF reader
```

### Step 4: Set Up Ollama and Download Models

This project uses `Ollama` to run the local language model.

```bash
# 1. Download and install the Ollama application from https://ollama.com

# 2. Launch the Ollama application. You should see a llama icon in your Mac's menu bar.

# 3. Pull the required model from the command line
ollama pull llama3.1:8b
```

### Step 5: Configure Nomic API Key

The Nomic embedding model requires a free API key.

```bash
# 1. Get a free API key by running the following command in your terminal.
#    It will print a URL for you to visit.
nomic login

# 2. Open the URL, log in, and copy the token it provides.

# 3. Paste the token back into the waiting terminal prompt.

# 4. For the application to run, you must set this key as an environment variable.
#    Replace 'your_key_here' with your actual key.
export NOMIC_API_KEY="your_key_here"
```
**Note:** You must run the `export` command in the same terminal session where you will run the application scripts.

## How to Run the Project

The project is split into two main scripts: `ingest.py` for data processing and `app.py` for the chat interface.

### 1. Run the Ingestion Pipeline (Run this First)

This script processes all source documents, intelligently extracts structured data into `dolly_facts.json`, and builds the `faiss_db` vector store.

```bash
# Ensure your virtual environment is active and the API key is set
python ingest.py
```
This process is computationally intensive and may take several minutes. Please wait for it to complete.

### 2. Launch the Chat Application

Once the ingestion is complete, you can start the interactive chat application.

```bash
# Ensure your virtual environment is active, the API key is set, and Ollama is running
streamlit run app.py
```
A new tab will open in your browser at `http://localhost:8501`. You can now start chatting with Dolly!

## Example Questions to Test

**Factual Questions (tests the `FactQueryEngine`):**
- Who is your husband?
- What are your children's names?
- When were you born?
- What is EWE?
- What was your thesis about?

**Narrative Questions (tests the Vector Store and Persona Layer):**
- How was your trip to Hawaii?
- Tell me about your passion for pottery.
- What did your wedding vows say?
- How did it feel to be the first clone?
