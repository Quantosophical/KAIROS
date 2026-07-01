from dataclasses import dataclass, field
from sampler import sample
from claim_extractor import extract_claims
from pressure_engine import compute_all_pressures
from zone_classifier import classify_all, print_field

@dataclass
class KAIROSField:
    query: str
    answer: str
    records: list[dict] = field(default_factory=list)
    fault_lines: list[dict] = field(default_factory=list)
    gradients: list[dict] = field(default_factory=list)
    solid: list[dict] = field(default_factory=list)
    total_claims: int
    N_samples: int

def build_kairos_field(query: str,
                       N: int = 5,
                       tau: float = 0.7,
                       vocab_size: int = 32000) -> KAIROSField:
    # Step 1 — Sample
    print("[ KAIROS ] Sampling " + str(N) + " times...")
    sample_set = sample(query, N=N, tau=tau)
    original_answer = sample_set.samples[0]
    print("[ KAIROS ] Primary answer captured.")
    print("")

    # Step 2 — Extract claims
    print("[ KAIROS ] Extracting claims...")
    claims = extract_claims(original_answer)
    print("[ KAIROS ] Found " + str(len(claims)) + " claims.")
    print("")
    
    if len(claims) == 0:
        print("[ KAIROS ] No claims extracted. Returning empty field.")
        return KAIROSField(
            query=query,
            answer=original_answer,
            records=[],
            fault_lines=[],
            gradients=[],
            solid=[],
            total_claims=0,
            N_samples=N
        )

    # Step 3 — Compute pressure for all claims
    print("[ KAIROS ] Computing uncertainty pressures...")
    print("(This runs LLM calls + embeddings. Takes 1-3 minutes.)")
    print("")
    records = compute_all_pressures(
        claims, original_answer, query, sample_set, vocab_size
    )
    print("")
    print("[ KAIROS ] Pressures computed.")
    print("")

    # Step 4 — Classify zones
    print("[ KAIROS ] Classifying zones...")
    classified = classify_all(records)
    print("[ KAIROS ] Classification complete.")
    print("")

    # Step 5 — Split into zone buckets
    fault_lines = [r for r in classified if r["zone"] == "FAULT LINE"]
    gradients   = [r for r in classified if r["zone"] == "GRADIENT"]
    solid       = [r for r in classified if r["zone"] == "SOLID"]

    # Step 6 — Build and return KAIROSField
    return KAIROSField(
        query        = query,
        answer       = original_answer,
        records      = classified,
        fault_lines  = fault_lines,
        gradients    = gradients,
        solid        = solid,
        total_claims = len(classified),
        N_samples    = N
    )

if __name__ == "__main__":
    query = "How does the human immune system fight a virus?"

    print("Running KAIROS on query:")
    print(query)
    print("")

    field = build_kairos_field(query, N=5)

    print_field(field.records)

    print("")
    print("=== FIELD SUMMARY ===")
    print("Query:        " + field.query[:60])
    print("Total claims: " + str(field.total_claims))
    print("Fault lines:  " + str(len(field.fault_lines)))
    print("Gradients:    " + str(len(field.gradients)))
    print("Solid:        " + str(len(field.solid)))
    print("N samples:    " + str(field.N_samples))
    print("")
    print("Field builder working.")
