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
            print("Response keys:", list(response.json().keys()))
            print("Full response sample:", str(response.json())[:500])
            
            # Extract answer text
            answer = resp_json.get("response", "")
            samples.append(answer)
            
            # Process logprobs
            logprobs_data = resp_json.get("logprobs", [])
            if isinstance(logprobs_data, list) and len(logprobs_data) > 0:
                token_dists = []
                for entry in logprobs_data:
                    token_dist = {}
                    if isinstance(entry, dict) and "top_logprobs" in entry:
                        top_logprobs = entry["top_logprobs"]
                        if isinstance(top_logprobs, list):
                            for item in top_logprobs:
                                if isinstance(item, dict) and "token" in item and "logprob" in item:
                                    try:
                                        prob = math.exp(item["logprob"])
                                        token_dist[item["token"]] = prob
                                    except (ValueError, OverflowError):
                                        pass
                    
                    # Normalize token distribution
                    if token_dist:
                        total_prob = sum(token_dist.values())
                        if total_prob > 0:
                            for tok in token_dist:
                                token_dist[tok] /= total_prob
                            token_dists.append(token_dist)
                        else:
                            token_dists.append({})
                    else:
                        token_dists.append({})
                
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
