import sys
from pressure_engine import UncertaintyRecord

# Configure stdout to support UTF-8 characters on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# Constants
PHI_S = 0.20   # Solid threshold
PHI_F = 0.60   # Fault Line threshold

def classify_zone(U: float,
                  phi_s: float = PHI_S,
                  phi_f: float = PHI_F) -> str:
    """
    Classifies a zone based on the uncertainty U.
    """
    if U < phi_s:
        return "SOLID"
    elif U < phi_f:
        return "GRADIENT"
    else:
        return "FAULT LINE"

def classify_record(record: UncertaintyRecord,
                    phi_s: float = PHI_S,
                    phi_f: float = PHI_F) -> dict:
    """
    Prints the zone assignment and returns a dict representation of the record
    with the zone classification added.
    """
    zone = classify_zone(record.U, phi_s, phi_f)
    print(f"Classified: '{record.claim}' -> {zone}")
    return {
        "claim":        record.claim,
        "H_e_norm":     record.H_e_norm,
        "G_norm":       record.G_norm,
        "Cons":         record.Cons,
        "inconsistency": record.inconsistency,
        "U":            record.U,
        "zone":         zone
    }

def classify_all(records: list[UncertaintyRecord],
                 phi_s: float = PHI_S,
                 phi_f: float = PHI_F) -> list[dict]:
    """
    Classifies all UncertaintyRecord objects.
    """
    results = []
    for record in records:
        classified = classify_record(record, phi_s, phi_f)
        results.append(classified)
    return results

def print_field(classified_records: list[dict]) -> None:
    """
    Prints a formatted field report for the classified records.
    """
    print("")
    print("=" * 70)
    print("  KAIROS UNCERTAINTY FIELD")
    print("=" * 70)
    
    # First print all FAULT LINE claims
    fault_lines = [r for r in classified_records if r["zone"] == "FAULT LINE"]
    if fault_lines:
        print("")
        print("  [FAULT LINE] Verify these before acting:")
        for r in fault_lines:
            print("  ⚠  " + r["claim"])
            print(f"     U={r['U']}  H={r['H_e_norm']}  G={r['G_norm']}  Cons={r['Cons']}")
            
    # Then print GRADIENT claims
    gradients = [r for r in classified_records if r["zone"] == "GRADIENT"]
    if gradients:
        print("")
        print("  [GRADIENT] Worth checking:")
        for r in gradients:
            print("  ◈  " + r["claim"])
            print(f"     U={r['U']}")
            
    # Then print SOLID claims
    solid = [r for r in classified_records if r["zone"] == "SOLID"]
    if solid:
        print("")
        print("  [SOLID] Trust these:")
        for r in solid:
            print("  ●  " + r["claim"])
            print(f"     U={r['U']}")
            
    # Footer
    print("")
    print("=" * 70)
    print(
        f"  {len(fault_lines)} FAULT LINE  |  "
        f"{len(gradients)} GRADIENT  |  "
        f"{len(solid)} SOLID  |  "
        f"{len(classified_records)} total claims"
    )
    print("=" * 70)
    print("")

if __name__ == "__main__":
    from pressure_engine import UncertaintyRecord

    test_records = [
        UncertaintyRecord(
            claim="The sky is blue",
            H_e_norm=0.02, G_norm=0.3, Cons=0.95,
            inconsistency=0.05, U=0.003),
        UncertaintyRecord(
            claim="Water boils at 100 degrees Celsius at sea level",
            H_e_norm=0.05, G_norm=0.7, Cons=0.88,
            inconsistency=0.12, U=0.004),
        UncertaintyRecord(
            claim="The dosage should be adjusted based on weight",
            H_e_norm=0.35, G_norm=0.85, Cons=0.62,
            inconsistency=0.38, U=0.113),
        UncertaintyRecord(
            claim="This drug interaction may cause cardiac arrest",
            H_e_norm=0.72, G_norm=0.94, Cons=0.41,
            inconsistency=0.59, U=0.400),
        UncertaintyRecord(
            claim="The contraindication applies in all cases",
            H_e_norm=0.88, G_norm=0.97, Cons=0.28,
            inconsistency=0.72, U=0.615),
    ]

    classified = classify_all(test_records)
    print_field(classified)
    print("Zone classifier working.")
