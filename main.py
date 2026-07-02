from field_builder import build_kairos_field
from zone_classifier import print_field

def run_kairos():
    print("")
    print("=" * 64)
    print("  K A I R O S")
    print("  Knowledge-Aware Inference with Realized")
    print("  Uncertainty Structure")
    print("=" * 64)
    print("  Type a question. KAIROS maps its uncertainty.")
    print("  Type 'quit' to exit.")
    print("=" * 64)
    print("")

    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("")
            print("KAIROS session ended.")
            break
            
        if query == "":
            continue
            
        print("")
        print("[ KAIROS ] Processing...")
        print("")
        
        try:
            field = build_kairos_field(query, N=3)
            print_field(field.records)
            
            print("Answer used for analysis:")
            print("")
            print(field.answer[:400] + "...")
            print("")
            
        except Exception as e:
            print("Error: " + str(e))
            print("Try again.")
            
        print("")

if __name__ == "__main__":
    run_kairos()
