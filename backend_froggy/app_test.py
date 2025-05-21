from flask import Flask, request, jsonify
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from together import Together
from flask_cors import CORS
import os

# === Settings ===
SANITY_API_BASE = "https://594hcrq0.api.sanity.io/v2025-04-14/data/query/production?query="
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# === Initialize clients ===
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/tmp/all-MiniLM-L6-v2"
)
client = Together(api_key=TOGETHER_API_KEY)

app = Flask(__name__)

# Configure CORS for production
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://aqualens.info"]
    }
})

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

def get_top_k(query, documents, query_type="projects", k=5):
    corpus_texts = []
    
    # Build corpus_texts depending on query_type
    if query_type == "projects":
        corpus_texts = [doc.get('projectName', '') + " " + doc.get('description', '') + " " + doc.get('fullText', '') + " " + doc.get('source', '') for doc in documents]
    elif query_type == "tags":
        corpus_texts = [doc.get('tagName', '') + " " + doc.get('description', '') for doc in documents]
    elif query_type == "locations":
        corpus_texts = [doc.get('locationName', '') + " " + doc.get('description', '') for doc in documents]
    elif query_type == "organizations":
        corpus_texts = [doc.get('orgName', '') + " " + doc.get('description', '') for doc in documents]

    # If there are no documents, return empty list
    if not corpus_texts:
        return []

    corpus_embeddings = embed_texts(corpus_texts)

    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k)
    
    # Return documents and their relevance scores
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):  # Ensure index is within range
            results.append({
                "document": documents[idx],
                "score": float(distances[0][i])  # Convert np.float32 to native Python float
            })
    
    # Sort by relevance (lower distance is more relevant)
    results.sort(key=lambda x: x["score"])
    
    # Return only the documents
    return [item["document"] for item in results]


def generate_answer(query, relevant_docs):
    # Generate context based on the documents returned
    context_parts = []
    
    for i, doc in enumerate(relevant_docs):
        if "projectName" in doc:
            context_parts.append(
                f"Document {i+1} (Project):\n"
                f"Project Name: {doc.get('projectName', 'N/A')}\n"
                f"Description: {doc.get('description', 'N/A')}\n"
                f"Full Text: {doc.get('fullText', '')[:800]}...\n"
                f"Tags: {', '.join(doc.get('tags', []))}\n"
                f"Location: {doc.get('location', 'N/A')}\n"
                f"Organization: {doc.get('companyOrganization', 'N/A')}\n"
                f"Source: {doc.get('source', 'N/A')}"
            )
        elif "tagName" in doc:
            context_parts.append(
                f"Document {i+1} (Tag):\n"
                f"Tag Name: {doc.get('tagName', 'N/A')}\n"
                f"Description: {doc.get('description', 'N/A')}"
            )
        elif "locationName" in doc:
            context_parts.append(
                f"Document {i+1} (Location):\n"
                f"Location Name: {doc.get('locationName', 'N/A')}\n"
                f"Description: {doc.get('description', 'N/A')}"
            )
        elif "orgName" in doc:
            context_parts.append(
                f"Document {i+1} (Organization):\n"
                f"Organization Name: {doc.get('orgName', 'N/A')}\n"
                f"Description: {doc.get('description', 'N/A')}"
            )
    
    # Join all context parts with double line breaks
    context = "\n\n".join(context_parts)
    
    # If no relevant documents were found, make it clear in the context
    if not relevant_docs:
        context = "No specific information about this query was found in our database."

    system_prompt = """You are Froggy, a helpful and knowledgeable AI agent specializing in water quality initiatives and research. Your responses must be based ONLY on the information provided in the context below. 

IMPORTANT INSTRUCTIONS:
1. ONLY use information that appears in the context documents provided.
2. If the context doesn't contain relevant information to answer the question, say "I don't have specific information about that in my database" rather than providing general knowledge.
3. DO NOT use any external knowledge or make up information that isn't in the provided context.
4. If only partial information is available, be clear about what you know from the context and what you don't know.
5. Always mention which document(s) in the context you're referencing in your answer.
6. Be concise and focus specifically on what the user is asking.

Your goal is to be accurate and truthful while only using information from the Sanity database."""

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
    # Get data from all sources to provide comprehensive results
    all_relevant_docs = []
    
    # Always search projects by default
    projects = query_sanity_data("projects")
    project_docs = get_top_k(user_question, projects, "projects")
    all_relevant_docs.extend(project_docs)
    
    # Check if there are specific mentions of tags, locations, or organizations
    query_lower = user_question.lower()
    
    # If the query mentions tags, add tag data
    if "tag" in query_lower or "category" in query_lower or "topic" in query_lower:
        tags = query_sanity_data("tags")
        tag_docs = get_top_k(user_question, tags, "tags")
        all_relevant_docs.extend(tag_docs)
    
    # If the query mentions locations, add location data
    if "location" in query_lower or "place" in query_lower or "region" in query_lower or "country" in query_lower:
        locations = query_sanity_data("locations")
        location_docs = get_top_k(user_question, locations, "locations")
        all_relevant_docs.extend(location_docs)
    
    # If the query mentions organizations, add organization data
    if "organization" in query_lower or "company" in query_lower or "institution" in query_lower:
        organizations = query_sanity_data("organizations")
        org_docs = get_top_k(user_question, organizations, "organizations")
        all_relevant_docs.extend(org_docs)
    
    # Limit to top 5 most relevant documents across all sources
    # Rerank all docs together by relevance
    if all_relevant_docs:
        corpus_texts = []
        for doc in all_relevant_docs:
            if "projectName" in doc:
                text = doc.get('projectName', '') + " " + doc.get('description', '') + " " + doc.get('fullText', '')
            elif "tagName" in doc:
                text = doc.get('tagName', '') + " " + doc.get('description', '')
            elif "locationName" in doc:
                text = doc.get('locationName', '') + " " + doc.get('description', '')
            elif "orgName" in doc:
                text = doc.get('orgName', '') + " " + doc.get('description', '')
            else:
                text = ""
            corpus_texts.append(text)
        
        corpus_embeddings = embed_texts(corpus_texts)
        index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)
        
        query_embedding = embed_texts([user_question])
        distances, indices = index.search(query_embedding, min(5, len(all_relevant_docs)))
        
        final_docs = [all_relevant_docs[i] for i in indices[0]]
    else:
        final_docs = []
    
    return generate_answer(user_question, final_docs)


# === API Endpoint ===
@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = answer_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Simple health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


# === Run server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)