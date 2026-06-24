import requests
import json

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def extract_claims(answer: str) -> list[str]:
    """
    Extracts individual factual claims from the provided answer text.
    Sends a prompt to Ollama to decompose the text into atomic, standalone sentences.
    """
    system_instruction = (
        "You are a precise claim extractor. Given a text, extract \n"
        "every individual factual claim as a separate item. \n"
        "Each claim must be:\n"
        "- A single, complete, standalone sentence\n"
        "- Not dependent on other claims to be understood\n"
        "- Containing exactly one assertion or fact\n"
        "   \n"
        "Return ONLY a numbered list. One claim per line. \n"
        "No preamble. No explanation. No extra text.\n"
        "Format: 1. [claim] 2. [claim] etc."
    )
    
    prompt = f"{system_instruction}\n\nText:\n{answer}"
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data.get("response", "")
        
        claims = []
        for line in response_text.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Remove the number prefix (e.g., "1. " or "2. ")
            if "." in line:
                parts = line.split(".", 1)
                if parts[0].isdigit():
                    line = parts[1].strip()
            
            if line:
                claims.append(line)
                
        return claims
        
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []

if __name__ == "__main__":
    test_answer = (
        "Photosynthesis is the process by which plants convert \n"
        "sunlight into glucose. This process occurs in the \n"
        "chloroplasts. It requires carbon dioxide and water as \n"
        "inputs. Oxygen is released as a byproduct. The process \n"
        "has two main stages: the light reactions and the \n"
        "Calvin cycle."
    )
    
    claims = extract_claims(test_answer)
    
    print("Extracted claims:")
    for idx, claim in enumerate(claims, 1):
        print(f"{idx}. {claim}")
    print(f"Total claims: {len(claims)}")
