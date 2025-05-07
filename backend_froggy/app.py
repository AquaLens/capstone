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
# Your existing backend logic remains the same...

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