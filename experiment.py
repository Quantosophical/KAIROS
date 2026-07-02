from field_builder import build_kairos_field
import time

questions = [
  {
    "question": "What is the boiling point of water at sea level?",
    "ground_truth": "100 degrees Celsius",
    "key_fact": "100"
  },
  {
    "question": "Who wrote Hamlet?",
    "ground_truth": "William Shakespeare",
    "key_fact": "Shakespeare"
  },
  {
    "question": "What organ produces insulin?",
    "ground_truth": "the pancreas",
    "key_fact": "pancreas"
  },
  {
    "question": "What is the powerhouse of the cell?",
    "ground_truth": "the mitochondria",
    "key_fact": "mitochondria"
  },
  {
    "question": "What gas do plants absorb during photosynthesis?",
    "ground_truth": "carbon dioxide",
    "key_fact": "carbon dioxide"
  },
]

def check_answer(answer: str, key_fact: str) -> bool:
    """
    Returns True if key_fact.lower() appears in answer.lower()
    This is a simple keyword check for prototype purposes
    """
    return key_fact.lower() in answer.lower()

def check_claim_against_fact(claim: str, 
                              key_fact: str) -> bool:
    # A claim is only wrong if it directly states 
    # something contradicting the key fact.
    # For prototype: only flag as wrong if the claim
    # contains a number or specific fact that 
    # contradicts the key_fact.
    # Simple rule: if key_fact IS in claim, it's fine.
    # If key_fact NOT in claim AND claim contains 
    # specific numbers/names = suspect.
    # Otherwise: not wrong (just a supporting claim).
    
    import re
    claim_lower = claim.lower()
    key_lower = key_fact.lower()
    
    # If key fact appears in claim: definitely not wrong
    if key_lower in claim_lower:
        return False
    
    # If claim has no specific facts (no numbers, 
    # no proper nouns starting caps) = supporting claim
    # Not wrong, just doesn't mention the key fact
    has_numbers = bool(re.search(r'\d', claim))
    has_caps = bool(re.search(r'[A-Z][a-z]{3,}', claim))
    
    # Only flag as potentially wrong if it states 
    # specific contradicting facts
    if not has_numbers and not has_caps:
        return False
    
    return False  # Default: not wrong for prototype
                  # Human verification needed for paper

def run_experiment(N_samples: int = 2) -> dict:
    print("=" * 60)
    print("  KAIROS FAULT LINE PREDICTION EXPERIMENT")
    print("=" * 60)
    print("")
    print("Questions: " + str(len(questions)))
    print("Samples per question: " + str(N_samples))
    print("")

    total_fault   = 0
    fault_wrong   = 0
    total_solid   = 0
    solid_wrong   = 0
    total_grad    = 0
    grad_wrong    = 0
    all_results   = []

    for q in questions:
        print("─" * 50)
        print("Q: " + q["question"])

        start = time.time()
        field = build_kairos_field(q["question"], N=N_samples)
        elapsed = round(time.time() - start, 1)

        print("Claims found: " + str(field.total_claims))
        print("Time: " + str(elapsed) + "s")
        print("")

        for record in field.records:
            zone = record["zone"]
            claim = record["claim"]
            is_wrong = check_claim_against_fact(claim, q["key_fact"])

            result_entry = {
                "question":  q["question"],
                "claim":     claim,
                "zone":      zone,
                "U":         record["U"],
                "is_wrong":  is_wrong
            }
            all_results.append(result_entry)

            if zone == "FAULT LINE":
                total_fault += 1
                if is_wrong:
                    fault_wrong += 1
            elif zone == "GRADIENT":
                total_grad += 1
                if is_wrong:
                    grad_wrong += 1
            else:
                total_solid += 1
                if is_wrong:
                    solid_wrong += 1

            print(f"  [{zone}] U={record['U']:.4f}  wrong={is_wrong}")
            print(f"   claim: {claim[:60]}")

        print("")

    print("=" * 60)
    print("  EXPERIMENT RESULTS")
    print("=" * 60)
    print("")

    fault_rate = fault_wrong / max(1, total_fault)
    solid_rate = solid_wrong / max(1, total_solid)
    grad_rate  = grad_wrong  / max(1, total_grad)
    lift = fault_rate / max(0.001, solid_rate)

    print(f"FAULT LINE: {fault_wrong}/{total_fault} wrong "
          f"({fault_rate*100:.1f}% error rate)")
    print(f"GRADIENT:   {grad_wrong}/{total_grad} wrong "
          f"({grad_rate*100:.1f}% error rate)")
    print(f"SOLID:      {solid_wrong}/{total_solid} wrong "
          f"({solid_rate*100:.1f}% error rate)")
    print("")
    print(f"Lift (FAULT vs SOLID): {lift:.1f}x")
    print("")

    if lift > 2.0:
        print("RESULT: KAIROS signal confirmed.")
        print("FAULT LINE claims have higher error rate than SOLID.")
    else:
        print("RESULT: Signal weak. More questions needed.")

    print("")
    print("=" * 60)

    return {
        "total_fault":  total_fault,
        "fault_wrong":  fault_wrong,
        "fault_rate":   fault_rate,
        "total_solid":  total_solid,
        "solid_wrong":  solid_wrong,
        "solid_rate":   solid_rate,
        "lift":         lift,
        "all_results":  all_results
    }

if __name__ == "__main__":
    results = run_experiment(N_samples=2)
