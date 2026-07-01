import requests
import json
import math

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

class SampleSet:
    """
    Represents a set of sampled answers and their corresponding token log probabilities.
    """
    def __init__(self, query: str, samples: list[str], logprobs: list[list[dict]]):
        self.query = query
        self.samples = samples
        self.logprobs = logprobs

def sample(query: str, N: int = 10, tau: float = 0.7) -> SampleSet:
    """
    Generates N samples from Ollama for the given query and extracts token logprobs.
    """
    samples = []
    logprobs = []
    
    for i in range(N):
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": query,
            "stream": False,
            "options": {
                "temperature": tau,
                "num_predict": 200
            },
            "logprobs": True,
            "top_k": 20
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            resp_json = response.json()
            
            # Extract answer text
            answer = resp_json.get("response", "")
            samples.append(answer)
            
            # Process logprobs
            logprob_data = resp_json.get("logprobs", [])
            
            # Handle dict format
            if isinstance(logprob_data, dict):
                logprob_data = logprob_data.get("content", [])

            # Handle list of token objects  
            token_dists = []
            if isinstance(logprob_data, list) and len(logprob_data) > 0:
                for token_entry in logprob_data:
                    dist = {}
                    # Try top_logprobs first
                    top = token_entry.get("top_logprobs", [])
                    if top:
                        for item in top:
                            tok = item.get("token", "")
                            lp  = item.get("logprob", -10)
                            if tok:
                                dist[tok] = math.exp(lp)
                    # Fallback: use the single token
                    if not dist:
                        tok = token_entry.get("token", "")
                        lp  = token_entry.get("logprob", -10)
                        if tok:
                            dist[tok] = math.exp(lp)
                    # Normalize
                    total = sum(dist.values())
                    if total > 0:
                        dist = {k: v/total for k, v in dist.items()}
                    token_dists.append(dist)
                
                logprobs.append(token_dists)
            else:
                logprobs.append([])
                print(f"Warning: logprobs not available for sample {i}")
                
        except Exception as e:
            # Handle failures gracefully by appending empty values
            samples.append("")
            logprobs.append([])
            print(f"Warning: logprobs not available for sample {i}. Exception: {e}")
            
    return SampleSet(query=query, samples=samples, logprobs=logprobs)

if __name__ == "__main__":
    query = "What causes rainbows to form?"
    result = sample(query, N=3, tau=0.7)
    
    print("Query:", query)
    print("Number of samples:", len(result.samples))
    
    for idx, sample_text in enumerate(result.samples):
        print(f"Sample {idx} (first 100 chars):")
        print(sample_text[:100])
        
    # Count of non-empty logprob lists
    non_empty_logprobs = sum(1 for lp in result.logprobs if lp)
    print("Logprobs available:", non_empty_logprobs)
    
    # Token positions in sample 0
    token_positions = len(result.logprobs[0]) if result.logprobs[0] else 0
    print("Token positions in sample 0:", token_positions)
