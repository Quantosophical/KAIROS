import re
from sampler import SampleSet
from gradient_engine import get_embedding, cosine_similarity

def extract_from_sample(sample_text: str, claim_text: str) -> str:
    """
    Splits sample_text into sentences and returns the sentence with highest
    cosine similarity to claim_text.
    """
    # Splits sample_text into sentences. Split on ". " and "? " and "! " and "\n"
    sentences = re.split(r'\. |\? |! |\n', sample_text)
    
    # Strip whitespace from each sentence and skip empty sentences
    cleaned_sentences = []
    for s in sentences:
        s_stripped = s.strip()
        if s_stripped:
            cleaned_sentences.append(s_stripped)
            
    if not cleaned_sentences:
        return sample_text[:200]
        
    claim_emb = get_embedding(claim_text)
    best_sentence = None
    best_sim = -2.0  # Cosine similarity range is [-1, 1]
    
    for sentence in cleaned_sentences:
        sent_emb = get_embedding(sentence)
        sim = cosine_similarity(sent_emb, claim_emb)
        if sim > best_sim:
            best_sim = sim
            best_sentence = sentence
            
    return best_sentence if best_sentence is not None else sample_text[:200]

def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamps value x between lo and hi.
    """
    return max(lo, min(hi, x))

def compute_consistency(claim_text: str, sample_set: SampleSet) -> float:
    """
    Computes consistency of a claim across a SampleSet based on semantic similarity
    of closest sentences in each sample.
    """
    extractions = []
    for sample_val in sample_set.samples:
        closest = extract_from_sample(sample_val, claim_text)
        extractions.append(closest)
        
    N = len(extractions)
    total_sim = 0.0
    pair_count = 0
    
    # Pre-compute embeddings for all extractions to avoid redundant computations
    extractions_embeddings = [get_embedding(ext) for ext in extractions]
    
    for n in range(N - 1):
        for k in range(n + 1, N):
            sim = cosine_similarity(extractions_embeddings[n], extractions_embeddings[k])
            total_sim += sim
            pair_count += 1
            
    if pair_count == 0:
        return 1.0
        
    cons = total_sim / pair_count
    return clamp(cons, 0.0, 1.0)

def compute_all_consistencies(claims: list[str], sample_set: SampleSet) -> dict:
    """
    Computes consistency scores for multiple claims.
    """
    results = {}
    for claim in claims:
        print("Computing consistency for: " + claim[:50] + "...")
        C = compute_consistency(claim, sample_set)
        results[claim] = C
    return results

if __name__ == "__main__":
    from sampler import sample
    
    query = "What is the speed of light and how was it measured?"
    
    print("Sampling 5 times...")
    ss = sample(query, N=5, tau=0.8)
    print("Samples collected:", len(ss.samples))
    print("")
    
    claims = [
        "The speed of light is approximately 299,792 kilometers per second",
        "Light speed was first measured by Ole Romer in 1676",
        "The speed of light is constant in a vacuum"
    ]
    
    print("Computing consistencies...")
    print("")
    results = compute_all_consistencies(claims, ss)
    
    print("=== CONSISTENCY RESULTS ===")
    for claim, C in results.items():
        zone = "STABLE" if C >= 0.8 else ("VARIABLE" if C >= 0.5 else "UNSTABLE")
        print(claim[:60] + "...")
        print("  Cons = " + str(round(C, 4)) + "  [" + zone + "]")
        print("")
        
    print("Consistency engine working.")
