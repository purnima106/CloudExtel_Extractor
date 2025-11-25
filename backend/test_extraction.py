"""
Test script to verify measurement extraction accuracy
"""
import os
from pathlib import Path
from app.services.measurement_extractor import extract_measurements

def test_sample_pdf():
    """Test with your sample PDF"""
    
    # Update this path to your actual PDF
    pdf_path = Path("../NMMC_812.pdf")
    
    # Alternative: use a specific backend_generated folder
    # pdf_path = Path("backend_generated/014a3ad9918c439a9f96e64a04ce0c80/NMMC_812.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please place your sample PDF as 'sample.pdf' in the backend directory")
        return
    
    print(f"📄 Testing extraction on: {pdf_path}")
    print("-" * 60)
    
    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Extract
    result = extract_measurements(pdf_bytes)
    
    # Display results
    print("\n✅ EXTRACTED DATA:")
    print("=" * 60)
    
    english = result.get("english", {})
    
    # Check critical fields (update these based on your actual PDF)
    checks = {
        "dn_number": "10/1182/2025",  # Should be from टे.क्र.
        "date": "9/10/2025",  # Should be from दि. (Devanagari converted)
        "document_number": None,  # Full नमुंमपा/परि... string
        "gst_number": "27AAALC0296J1Z4",
        "pan_number": "AAALC0296J",
        "total_amount": None,  # Update with expected value
    }
    
    print("\n🔍 CRITICAL FIELDS:")
    for field, expected in checks.items():
        actual = english.get(field, "NOT FOUND")
        if expected is None:
            status = "ℹ️"
            print(f"{status} {field}: {actual}")
        else:
            status = "✅" if str(actual) == str(expected) else "❌"
            print(f"{status} {field}: {actual} (expected: {expected})")
    
    print("\n📊 ALL EXTRACTED FIELDS:")
    for key, value in english.items():
        print(f"  • {key}: {value}")
    
    print("\n📋 TABLE ROWS:")
    table_rows = result.get("table_rows", [])
    if table_rows:
        for i, row in enumerate(table_rows, 1):
            print(f"  Row {i}: {row}")
    else:
        print("  No table rows extracted")
    
    print("\n" + "=" * 60)
    print(f"Total fields extracted: {len(english)}")
    print(f"Table rows extracted: {len(table_rows)}")
    
    # Save results
    import json
    output_file = "extraction_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in environment")
        print("Please set it in your .env file or environment variables")
    else:
        test_sample_pdf()

