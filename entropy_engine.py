import math
from sampler import SampleSet

def shannon_entropy(distribution: dict) -> float:
    """
    Computes Shannon entropy H = -sum(p * log2(p)) for a token distribution.
    Convention: if p == 0 or p <= 0, skip it (0 * log2(0) = 0).
    Returns H as a float, or 0.0 if the distribution is empty.
    """
    if not distribution:
        return 0.0
    
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log2(p)
    return h

def compute_mixture(sample_set: SampleSet, t: int) -> dict:
    """
    Computes the average distribution across all N samples at token position t.
    mixture[token] = (1/N) * sum over n of p_n(token at t).
    Only includes samples where t < len(sample_set.logprobs[n]).
    If no samples have data at position t, returns an empty dict.
    Normalizes the result so values sum to 1.0.
    """
    N = len(sample_set.logprobs)
    if N == 0:
        return {}
    
    mixture = {}
    valid_count = 0
    for lp in sample_set.logprobs:
        if t < len(lp):
            valid_count += 1
            token_dist = lp[t]
            for token, prob in token_dist.items():
                mixture[token] = mixture.get(token, 0.0) + prob
                
    if valid_count == 0 or not mixture:
        return {}
        
    # Multiply by 1/N
    for token in mixture:
        mixture[token] /= N
        
    # Normalize the result so values sum to 1.0
    total_prob = sum(mixture.values())
    if total_prob > 0:
        for token in mixture:
            mixture[token] /= total_prob
    else:
        return {}
        
    return mixture

def compute_H_e_norm(sample_set: SampleSet, t: int, vocab_size: int = 32000) -> float:
    """
    Computes normalized epistemic entropy at token position t.
    """
    logprobs_list = sample_set.logprobs
    print("DEBUG logprobs_list length:", len(logprobs_list))
    if len(logprobs_list) > 0:
        print("DEBUG sample 0 length:", len(logprobs_list[0]))
        if len(logprobs_list[0]) > 0:
            print("DEBUG token 0 type:", 
                  type(logprobs_list[0][0]))
            print("DEBUG token 0 value:", 
                  str(logprobs_list[0][0])[:200])
                  
    # 1. mixture = compute_mixture(sample_set, t)
    mixture = compute_mixture(sample_set, t)
    
    # 2. If mixture is empty, return 0.0
    if not mixture:
        return 0.0
        
    # 3. H_total = shannon_entropy(mixture)
    H_total = shannon_entropy(mixture)
    
    # 4. Compute H_aleatoric
    h_aleatoric_sum = 0.0
    valid_count = 0
    for lp in sample_set.logprobs:
        if t < len(lp):
            H_n = shannon_entropy(lp[t])
            h_aleatoric_sum += H_n
            valid_count += 1
            
    if valid_count == 0:
        return 0.0
        
    H_aleatoric = h_aleatoric_sum / valid_count
    
    # 5. H_e = H_total - H_aleatoric
    H_e = H_total - H_aleatoric
    
    # 6. H_e = max(0.0, H_e)
    H_e = max(0.0, H_e)
    
    # 7. H_max = log2(vocab_size)
    H_max = math.log2(vocab_size)
    
    # 8. If H_max == 0, return 0.0
    if H_max == 0:
        return 0.0
        
    # 9. H_e_norm = H_e / H_max
    H_e_norm = H_e / H_max
    
    # 10. Return clamp(H_e_norm, 0.0, 1.0)
    return max(0.0, min(1.0, H_e_norm))

def claim_entropy(claim_text: str, 
                  answer: str,
                  sample_set: SampleSet,
                  vocab_size: int = 32000) -> float:
    """
    Finds the approximate token position of claim_text in the answer and
    returns the maximum H_e_norm found across the estimated token range.
    """
    char_pos = answer.find(claim_text)
    
    if char_pos == -1:
        start_token = 0
        end_token = 0
    else:
        # Estimates start token = (char position of claim in answer) // 4
        start_token = char_pos // 4
        # Estimates end token = start token + (len(claim_text) // 4) + 1
        end_token = start_token + (len(claim_text) // 4) + 1
        
    # Clamp start and end to valid range of sample_set.logprobs[0]
    if not sample_set.logprobs or not sample_set.logprobs[0]:
        return 0.0
        
    max_idx = len(sample_set.logprobs[0]) - 1
    start_token = max(0, min(max_idx, start_token))
    end_token = max(0, min(max_idx, end_token))
    
    if start_token > end_token:
        start_token, end_token = end_token, start_token
        
    max_h_e_norm = 0.0
    for t in range(start_token, end_token + 1):
        h_e = compute_H_e_norm(sample_set, t, vocab_size)
        if h_e > max_h_e_norm:
            max_h_e_norm = h_e
            
    return max_h_e_norm

if __name__ == "__main__":
    from sampler import sample
    
    query = "What is the boiling point of water and why does altitude affect it?"
    print("Sampling... (this takes a moment)")
    result = sample(query, N=5, tau=0.8)
    print("Samples collected:", len(result.samples))
    
    # Test shannon_entropy on a made-up distribution
    test_dist = {"cat": 0.5, "dog": 0.3, "bird": 0.2}
    print("Shannon entropy test:", shannon_entropy(test_dist))
    print("Expected: ~1.485 bits")
    
    # Test compute_H_e_norm at token position 5
    H = compute_H_e_norm(result, t=5)
    print("H_e_norm at t=5:", H)
    
    # Test claim_entropy on two claims
    claim1 = "Water boils at 100 degrees Celsius"
    claim2 = "altitude affects the boiling point"
    h1 = claim_entropy(claim1, result.samples[0], result)
    h2 = claim_entropy(claim2, result.samples[0], result)
    
    print("Claim 1 entropy:", h1, "(", claim1, ")")
    print("Claim 2 entropy:", h2, "(", claim2, ")")
    
    if isinstance(h1, float) and 0.0 <= h1 <= 1.0 and isinstance(h2, float) and 0.0 <= h2 <= 1.0:
        print("Entropy engine working.")
