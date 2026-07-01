from field_builder import build_kairos_field
import time

questions = [
  {
    "question": "What is the boiling point of water at sea level?",
    "ground_truth": "100 degrees Celsius or 212 degrees Fahrenheit",
    "key_fact": "100"
  },
  {
    "question": "Who wrote the play Hamlet?",
    "ground_truth": "William Shakespeare",
    "key_fact": "Shakespeare"
  },
  {
    "question": "What is the speed of light in a vacuum?",
    "ground_truth": "approximately 299,792 kilometers per second",
    "key_fact": "299"
  },
  {
    "question": "What organ produces insulin in the human body?",
    "ground_truth": "the pancreas",
    "key_fact": "pancreas"
  },
  {
    "question": "What is the chemical formula for water?",
    "ground_truth": "H2O",
    "key_fact": "H2O"
  },
  {
    "question": "In what year did World War Two end?",
    "ground_truth": "1945",
    "key_fact": "1945"
  },
  {
    "question": "What is the largest planet in our solar system?",
    "ground_truth": "Jupiter",
    "key_fact": "Jupiter"
  },
  {
    "question": "What gas do plants absorb during photosynthesis?",
    "ground_truth": "carbon dioxide",
    "key_fact": "carbon dioxide"
  },
  {
    "question": "How many bones are in the adult human body?",
    "ground_truth": "206 bones",
    "key_fact": "206"
  },
  {
    "question": "What is the powerhouse of the cell?",
    "ground_truth": "the mitochondria",
    "key_fact": "mitochondria"
  },
]

def check_answer(answer: str, key_fact: str) -> bool:
    """
    Returns True if key_fact.lower() appears in answer.lower()
    This is a simple keyword check for prototype purposes
    """
    return key_fact.lower() in answer.lower()

def check_claim_against_fact(claim: str, key_fact: str) -> bool:
    """
    Returns True if the claim appears to CONTRADICT the key fact
    Simple heuristic: if key_fact.lower() NOT in claim.lower()
    AND claim is longer than 10 characters return True.
    """
    if key_fact.lower() not in claim.lower() and len(claim) > 10:
        return True
    return False

def run_experiment(N_samples: int = 3) -> dict:
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
    results = run_experiment(N_samples=3)
