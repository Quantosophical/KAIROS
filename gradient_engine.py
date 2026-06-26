import requests
import math
from sentence_transformers import SentenceTransformer
import numpy as np

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """
    Returns the sentence embedding for the given text.
    """
    return EMBEDDING_MODEL.encode(text)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(max(-1.0, min(1.0, similarity)))

def semantic_distance(text_a: str, text_b: str) -> float:
    """
    Computes the semantic distance between two texts.
    0.0 means identical meaning, 1.0 means completely different.
    """
    sim = cosine_similarity(get_embedding(text_a), get_embedding(text_b))
    return float(max(0.0, min(1.0, 1.0 - sim)))

def perturb_answer(query: str, claim_text: str) -> str:
    """
    Generates a perturbed answer by asking the model to assume the claim is false.
    """
    prompt = (
        f"Answer the following question.\n"
        f"Important assumption: The following statement \n"
        f"is FALSE: {claim_text}\n"
        f"Build your entire answer assuming this is wrong.\n"
        f"\n"
        f"Question: {query}"
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data["response"])
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return query

def compute_gradients(claims: list[str],
                      original_answer: str,
                      query: str) -> dict:
    """
    Computes gradients for each claim indicating structural importance.
    """
    raw_gradients = {}
    for claim in claims:
        print("Computing gradient for: " + claim[:50] + "...")
        perturbed = perturb_answer(query, claim)
        G_raw = semantic_distance(original_answer, perturbed)
        raw_gradients[claim] = G_raw
        
    normalized = {}
    if not raw_gradients:
        return normalized
        
    max_G = max(raw_gradients.values())
    for claim in claims:
        if max_G == 0.0:
            normalized[claim] = 0.0
        else:
            normalized[claim] = raw_gradients[claim] / max_G
            
    return normalized

if __name__ == "__main__":
    query = "What is photosynthesis and where does it occur?"
    
    original_answer = (
        "Photosynthesis is the process by which plants convert "
        "sunlight into glucose. This process occurs in the "
        "chloroplasts. It requires carbon dioxide and water. "
        "Oxygen is released as a byproduct."
    )
    
    claims = [
        "Photosynthesis converts sunlight into glucose",
        "This process occurs in the chloroplasts",
        "Oxygen is released as a byproduct"
    ]
    
    print("Testing semantic distance:")
    d = semantic_distance(original_answer, original_answer)
    print("Distance from itself:", d, "(should be ~0.0)")
    
    print("")
    print("Computing gradients (makes LLM calls, takes time)...")
    gradients = compute_gradients(claims, original_answer, query)
    
    print("")
    print("=== GRADIENT RESULTS ===")
    for claim, G in gradients.items():
        print(claim[:60] + " -> G_norm: " + str(round(G, 4)))
    
    print("")
    max_claim = max(gradients, key=gradients.get)
    print("Most structural claim: " + max_claim)
    print("Gradient engine working.")
