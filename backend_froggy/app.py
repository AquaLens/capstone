from flask import Flask, request, jsonify
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from together import Together
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Settings ===
SANITY_API_BASE = "https://594hcrq0.api.sanity.io/v2025-04-14/data/query/production?query="
TOGETHER_API_KEY = "ad48643b4e8a097dc24b2ff7a0d6cc312e4f74d5231a6cebe66e71173dced2da"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# === Initialize clients ===
app = Flask(__name__)

# Configure CORS properly with explicit options
CORS(app, resources={
    r"/*": {  # Apply to ALL routes
        "origins": "*",  # In production, replace with your specific origin
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Lazy load models only when needed to save memory
embedder = None
client = None

def get_embedder():
    global embedder
    if embedder is None:
        logger.info("Initializing sentence transformer model...")
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    return embedder

def get_client():
    global client
    if client is None:
        logger.info("Initializing Together client...")
        try:
            client = Together(api_key=TOGETHER_API_KEY)
            logger.info("Client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing client: {str(e)}")
            raise
    return client


# === Backend logic ===
def query_sanity_data(query_type="projects"):
    # Define different queries for different types of data
    queries = {
        "projects": '*[_type == "projects"]{projectName, description, fullText, tags, location, companyOrganization, source}',
        "tags": '*[_type == "tags"]{tagName, description}',
        "locations": '*[_type == "locations"]{locationName, description}',
        "organizations": '*[_type == "organizations"]{orgName, description}',
    }

    query = queries.get(query_type, queries["projects"])  # Default to projects if query_type is unknown
    url = SANITY_API_BASE + requests.utils.quote(query)
    res = requests.get(url)
    res.raise_for_status()
    return res.json()['result']


def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

def get_top_k(query, documents, query_type="projects", k=3):
    corpus_texts = []
    
    # Build corpus_texts depending on query_type
    if query_type == "projects":
        corpus_texts = [doc['description'] + "\n" + doc['fullText'] + "\n" + doc['source'] for doc in documents]
    elif query_type == "tags":
        corpus_texts = [doc['tagName'] + "\n" + doc['description'] for doc in documents]
    elif query_type == "locations":
        corpus_texts = [doc['locationName'] + "\n" + doc['description'] for doc in documents]
    elif query_type == "organizations":
        corpus_texts = [doc['orgName'] + "\n" + doc['description'] for doc in documents]

    corpus_embeddings = embed_texts(corpus_texts)

    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]


def generate_answer(query, relevant_docs, query_type="projects"):
    # Generate context based on the documents returned
    context = "\n\n".join(
        f"Name: {doc.get('projectName', doc.get('tagName', doc.get('locationName', doc.get('orgName', 'N/A'))))}\n"
        f"Description: {doc.get('description', 'N/A')}\n"
        f"Full Text: {doc.get('fullText', '')[:1000]}..."
        for doc in relevant_docs
    )

    # Adjust the system prompt to avoid the 'projects' assumption
    system_prompt = """You are Froggy, a helpful and knowledgeable AI agent specializing in water quality initiatives and research. My goal is to guide you by answering questions, providing insights, and helping you explore water-related topics such as research projects, locations, organizations, and more.

I will adjust my responses based on your interest:
- If you're curious about specific research, I can provide detailed explanations.
- If you're looking to explore water-related topics like locations, organizations, or technologies, I can help with that too.
- I'll keep my responses friendly, clear, and concise, and tailor them to your level of expertise.

Please feel free to ask anything about water research, and I'll be happy to assist you."""

    prompt = f"""Context:
{context}

User question: {query}
Answer:"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()



def answer_question(user_question):
    # First, classify the query (e.g., is it about projects, tags, locations, etc.?)
    if "project" in user_question.lower():
        query_type = "projects"
    elif "tag" in user_question.lower():
        query_type = "tags"
    elif "location" in user_question.lower():
        query_type = "locations"
    elif "organization" in user_question.lower():
        query_type = "organizations"
    else:
        query_type = None  # No query type if unsure

    if query_type:
        documents = query_sanity_data(query_type)
        top_docs = get_top_k(user_question, documents, query_type)
        return generate_answer(user_question, top_docs, query_type)
    else:
        return generate_answer(user_question, [], "")  # No documents and no query type



# === API Endpoints ===
@app.route("/", methods=["GET"])
def root():
    """Root endpoint that just confirms the API is running"""
    logger.info("Root endpoint accessed")
    return jsonify({
        "status": "API is running",
        "message": "Welcome to the Froggy API",
        "endpoints": {
            "/": "This help message",
            "/health": "Health check endpoint",
            "/api/ask": "Main API endpoint for asking questions (POST)",
            "/debug": "Debugging information"
        }
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "ok"}), 200

@app.route("/debug", methods=["GET"])
def debug_info():
    """Provides debugging information"""
    logger.info("Debug endpoint accessed")
    return jsonify({
        "routes": [str(rule) for rule in app.url_map.iter_rules()],
        "request_headers": dict(request.headers),
        "flask_env": app.config.get("ENV", "production"),
        "cors_enabled": "Yes",
        "python_version": platform.python_version() if platform else "Unknown"
    }), 200

@app.route("/api/ask", methods=["POST", "OPTIONS"])
def ask():
    """Main API endpoint for asking questions"""
    # Handle OPTIONS requests explicitly for CORS preflight
    if request.method == "OPTIONS":
        logger.info("OPTIONS request to /api/ask")
        return "", 200
    
    logger.info("POST request to /api/ask")
    try:
        data = request.json
        logger.info(f"Request data: {data}")
        
        question = data.get("question", "")
        if not question:
            logger.warning("No question provided")
            return jsonify({"error": "No question provided"}), 400

        # Simple echo for testing
        if question.lower() == "test":
            return jsonify({"answer": "Test successful! The API is working correctly."}), 200
            
        logger.info(f"Processing question: {question}")
        answer = "This is a placeholder response while the backend is being debugged."
        # In production, uncomment to use your actual answer_question function
        # answer = answer_question(question)
        
        logger.info("Successfully generated answer")
        return jsonify({"answer": answer}), 200
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# === Run server ===
if __name__ == "__main__":
    # Import platform only when needed to avoid issues
    import platform
    logger.info(f"Starting Flask server on port 10000")
    app.run(host="0.0.0.0", port=10000, debug=True)