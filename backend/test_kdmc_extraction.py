import os
from pathlib import Path
from app.services.measurement_extractor_kdmc import extract_measurements_kdmc
import json

def test_kdmc_pdf():
    # The long filename from the user's project folder
    pdf_name = "DN1_164_Mumbai_Coverage_Route13_KDMC_Kalyan Division_328 Mtrs (1).pdf"
    # It's in the parent directory relative to backend/
    pdf_path = Path(f"../{pdf_name}")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"📄 Testing KDMC extraction on: {pdf_name}")
    print("-" * 60)
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = extract_measurements_kdmc(pdf_bytes)
    
    print("\n✅ EXTRACTED DATA:")
    print("=" * 60)
    
    english = result.get("english", {})
    
    print("\n📊 FIELDS:")
    for key, value in english.items():
        print(f"  • {key}: {value}")
    
    print("\n📋 TABLE ROWS:")
    table_rows = result.get("table_rows", [])
    if table_rows:
        for i, row in enumerate(table_rows, 1):
            print(f"  Row {i}: {row}")
    else:
        print("  No table rows extracted")

    # Save for inspection
    with open("kdmc_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found")
    else:
        test_kdmc_pdf()
