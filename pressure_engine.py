from dataclasses import dataclass
from sampler import SampleSet


@dataclass
class UncertaintyRecord:
    claim: str
    H_e_norm: float   # epistemic entropy component
    G_norm: float     # structural gradient component
    Cons: float       # consistency score
    inconsistency: float  # 1 - Cons
    U: float          # final uncertainty pressure

def compute_pressure(H_e_norm: float,
                     G_norm: float,
                     Cons: float) -> float:
    inconsistency = 1.0 - Cons
    U = H_e_norm * G_norm * inconsistency
    U = max(0.0, min(1.0, U))  # clamp to [0,1]
    return U

def compute_all_pressures(claims: list[str],
                          original_answer: str,
                          query: str,
                          sample_set: SampleSet,
                          vocab_size: int = 32000) -> list[UncertaintyRecord]:
    """
    Computes uncertainty pressure for all claims by combining epistemic entropy,
    structural gradients, and consistency scores.
    """
    from entropy_engine import claim_entropy
    from gradient_engine import compute_gradients
    from consistency_engine import compute_consistency

    # Step 1 — Compute gradients for all claims at once:
    print("Computing gradients...")
    gradients = compute_gradients(claims, original_answer, query)

    # Step 2 — For each claim, compute all three components and assemble the UncertaintyRecord:
    records = []
    
    for claim in claims:
        print("Scoring: " + claim[:50] + "...")
        
        H = claim_entropy(claim, original_answer,
                          sample_set, vocab_size)
        
        G = gradients.get(claim, 0.0)
        
        C = compute_consistency(claim, sample_set)
        
        U = compute_pressure(H, G, C)
        
        record = UncertaintyRecord(
            claim         = claim,
            H_e_norm      = round(H, 4),
            G_norm        = round(G, 4),
            Cons          = round(C, 4),
            inconsistency = round(1.0 - C, 4),
            U             = round(U, 4)
        )
        
        records.append(record)
        
    return records

if __name__ == "__main__":
    from sampler import sample

    query = "What causes lightning and how does thunder form?"

    original_answer = (
        "Lightning is caused by the buildup of electrical charge "
        "in storm clouds. When the charge difference becomes large "
        "enough, electricity discharges between the cloud and ground. "
        "Thunder is the sound caused by the rapid expansion of air "
        "heated by the lightning bolt. Light travels faster than "
        "sound, which is why we see lightning before hearing thunder."
    )

    claims = [
        "Lightning is caused by electrical charge buildup in storm clouds",
        "Electricity discharges between cloud and ground",
        "Thunder is caused by rapid expansion of heated air",
        "Light travels faster than sound"
    ]

    print("Sampling...")
    ss = sample(query, N=5, tau=0.7)
    print("Samples collected:", len(ss.samples))
    print("")

    print("Computing all pressures...")
    records = compute_all_pressures(claims, original_answer,
                                    query, ss)
    print("")
    print("=== PRESSURE RESULTS ===")
    print(f"{'Claim':<45} {'H':>6} {'G':>6} {'C':>6} {'U':>6}")
    print("-" * 75)

    for record in records:
        print(f"{record.claim[:44]:<45} "
              f"{record.H_e_norm:>6.4f} "
              f"{record.G_norm:>6.4f} "
              f"{record.Cons:>6.4f} "
              f"{record.U:>6.4f}")

    print("")
    most_uncertain = max(records, key=lambda r: r.U)
    print("Highest pressure claim: " + most_uncertain.claim)
    print("U = " + str(most_uncertain.U))
    print("")
    print("Pressure engine working.")
