import fitz  # PyMuPDF
import camelot
import re
import datetime
import numpy as np
from pdf2image import convert_from_path
import cv2
import pytesseract
from PIL import Image
import os
import hashlib
import time

ALL_HEADERS = [
    "Intercity/Intracity- Deployment Intercity/intracity- O&M FTTH- Deployment FTTH-O&M",
    "BUSINESS UNIT",
    "Circle",
    "City",
    "Demand Note Reference number",
    "LM/BB/FTTH",
    "Type (UG/OH)",
    "Capping/Non Capping",
    "UG TYPE( HDD/ OT/ MICROTRENCHING)",
    "Road Types - CC/BT/TILES/ Normal Soil/kacha",
    "HDD - Number of Pits",
    "OH (EB Poles/MC Poles/Own Poles)",
    "NO OF POLES",
    "RAILWAY CROSSING/ PIPELINE CROSSING( No of crossing)",
    "GO RATE",
    "PREVIOUS DN RATE",
    "Rate/mtr- Current DN (UG/OH)",
    "Annual Rate/Pole( current DN)",
    "HDD(PIT RATE)",
    "Section Length (Mtr.)",
    "Total Route (MTR)",
    "RAILWAY/ PIPELINE/ EACH CROSSING RATE",
    "Reason (Current rate is more than GO or Previous DN)",
    "Annual Lease/ rent amount",
    "Renewal Lease/Rent date",
    "Not part of capping (License Fee/Rental Payment /Way Leave charges etc.)",
    "Covered under capping (Restoration Charges, admin, registration etc.)",
    "Non Refundable Cost (Amount to process for payment should be sum of 'A' + 'B' + 'AA')",
    "Cost type with Cost Breakup EG.. - PROCESING FEES/ SUPERVISOIN CHARGE/ ADMIN FEES/ LICENSE FEES etc etc.",
    "GST Amount",
    "BG Amount",
    "SD Amount",
    "ROW APPLICATION  DATE",
    "Demand Note Date",
    "DN RECEIVED FROM PARTNER/AUTHORITY- DATE",
    "Difference from, DN date  - DN Sent to Central team (ARTL)",
    "REASON FOR DELAY (>2 DAYS)",
    "Total DN Amount ( NON REFUNDABLE+SD+ BG+ GST) To be filled by helpdesk team",
    "Supplier Code( if team have) To be filled by helpdesk team",
    "Supplier site name( if team have) To be filled by helpdesk team",
    "Locator Code (material)",
    "Authority( email address)",
    "Authority",
    "BENEFICIERY NAME",
    "Mode of payment(DD/ONLINE-URL/ONLINE-NEFT/BHARATKOSH",
    "EXECUTION PARTNER NAME",
    "Payable (Authority) Location",
    "Printing Location",
    "PO No.",
    "Business NFA NUMBER (Approved CAF) To be filled by helpdesk team",
    "Route Name(As per CWIP)",
    "Section Name for ROW(As per CWIP)",
    "NSG ID(As per CWIP)/CWO NO.",
    "Total Amount as per capping MB(Partner Scope)",
    "Cost type(restoration/ supervison/ agency changes/ admin etc)",
    "Total Amount as per capping MB(Not in Partner Scope)",
    "Cost type (way leave charges/ rent/ license etc)",
    "Permission Type (Primary/ Secondary)",
    "Additional Remarks",
]

STATIC_VALUES = {
    "Intercity/Intracity- Deployment Intercity/intracity- O&M FTTH- Deployment FTTH-O&M": "Intercity/Intracity - Deployment",
    "BUSINESS UNIT": "TNL-FF-Maharashtra",
    "Circle": "MUM",
    "City": "MUM",
    "Capping/Non Capping": "Non capping",
    "Cost type with Cost Breakup EG.. - PROCESING FEES/ SUPERVISOIN CHARGE/ ADMIN FEES/ LICENSE FEES etc etc.": "Restoration Charges",
    "Authority( email address)": "dmc_zone_@nmmc.gov.in",
    "Authority": "Navi Mumbai Municipal Corporation",
    "BENEFICIERY NAME": "Navi Mumbai Municipal Corporation",
    "Mode of payment(DD/ONLINE-URL/ONLINE-NEFT/BHARATKOSH": "DD",
    "EXECUTION PARTNER NAME": "Excel Telesonic India Private Limited",
    "Payable (Authority) Location": "Navi Mumbai",
    "Cost type(restoration/ supervison/ agency changes/ admin etc)": "Restoration Charges",
    "Permission Type (Primary/ Secondary)": "Primary",
    "Type (UG/OH)": "UG",
}

# Print the Tesseract executable path for user reference
print('Tesseract executable path:', pytesseract.pytesseract.tesseract_cmd)

# Explicitly set the Tesseract executable path to the correct location
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\SashwatRavishankar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# --- Extraction Functions (blank, unified names) ---

def extract_section_length(text):
    """
    Robustly extract the section length (e.g., 825.00) from context sentences in Marathi/English.
    Look for lines with 'लांबी', 'length', or sentences mentioning cable laying, and extract the first number with decimals (3-6 digits, e.g., 825.00).
    Return as a string, or empty if not found.
    """
    import re
    # 1. Look for lines with 'लांबी', 'length', or cable laying context
    for line in text.splitlines():
        if re.search(r'लांबी|length|केबल|optical|फायबर', line, re.IGNORECASE):
            match = re.search(r'(\d{2,4}\.\d{1,3})', line)
            if match:
                return match.group(1)
    # 2. Fallback: any number with decimals in the text
    match = re.search(r'(\d{2,4}\.\d{1,3})', text)
    if match:
        return match.group(1)
    return ''



def extract_difference_days(received_date):
    """Calculates difference in days from received date to today."""
    pass



def extract_ground_rent(text, pdf_path=None):
    """Returns the same value as not_part_of_capping (Land Rent)."""
    return extract_not_part_of_capping_from_text(text, pdf_path)

def extract_administrative_charge(text):
    """Extracts administrative charge from text."""
    pass

def extract_multiplication_factor(text):
    """Extracts multiplication factor from text."""
    # Table extraction removed. Implement only text-based extraction if needed.
    pass

def extract_surface_wise_length(text):
    """Extracts surface-wise length from text."""
    # Table extraction removed. Implement only text-based extraction if needed.
    pass

def normalize_key(key):
    return ''.join(key.lower().split())

def extract_ri_from_table_rows(rows):
    """
    Robustly extract the RI (Reinstallation) amount from table rows.
    1. Find a row where any cell contains 'Reinstallati' (case-insensitive, ignoring whitespace/newlines).
    2. In that row, extract all numbers from the cell(s) containing 'Reinstallati' and the next 2 cells to the right.
    3. Prefer numbers that are 5 or 6 digits and end with .00, or are repeated in the row.
    4. If not found, fallback to the previous logic (largest number in the row).
    """
    import re
    for row in rows:
        for idx, cell in enumerate(row):
            if cell and re.search(r'Reinstallati', str(cell).replace('\n', ''), re.IGNORECASE):
                # Gather numbers from this cell and next 2 cells to the right
                candidates = []
                for offset in range(0, 3):
                    if idx + offset < len(row):
                        c = row[idx + offset]
                        if c:
                            found = re.findall(r'\d{5,6}(?:\.\d{2})?', str(c).replace(',', ''))
                            candidates.extend(found)
                # Prefer numbers ending with .00
                for n in candidates:
                    if n.endswith('.00'):
                        return n
                # If not found, pick the most common candidate
                if candidates:
                    from collections import Counter
                    most_common = Counter(candidates).most_common(1)[0][0]
                    return most_common
                # Fallback: largest number in the row
                all_nums = re.findall(r'\d{5,6}(?:\.\d{2})?', ' '.join([str(x) for x in row if x]))
                if all_nums:
                    return max(all_nums, key=lambda x: float(x))
    return ''

# --- Unified Extraction Function ---
def extract_nmmc_all_fields(pdf_path):
    """
    Extracts all possible fields from the NMMC DN using robust OCR logic.
    Returns a dictionary with all fields that might be needed for any output table.
    """
    import re
    raw_text = ''
    pages = []
    if pdf_path:
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            for page_num, pil_img in enumerate(pages):
                temp_img_path = f"_nmmc_tmp_page_{page_num+1}.png"
                pil_img.save(temp_img_path, format="PNG")
                # OCR the image
                img = np.array(pil_img.convert('L'))
                text = pytesseract.image_to_string(img, config='--psm 6', lang='mar+eng')
                raw_text += '\n' + text
        except Exception as e:
            return {}
    # Table extraction for robust fields
    table_rows = []
    try:
        from extract_table_opencv import extract_table_from_image
        for page_num, pil_img in enumerate(pages):
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_img_path = tmp.name
                pil_img.save(temp_img_path, format="PNG")
            try:
                table = extract_table_from_image(temp_img_path)
                if hasattr(table, 'values'):
                    rows = table.values.tolist()
                else:
                    rows = table
                if rows is not None and len(rows) > 0:
                    table_rows.extend(rows)
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
    except Exception as e:
        pass
    # Robust field extraction
    total_dn_amount = extract_total_dn_amount_from_text(raw_text) or ''
    # --- Use table-based RI extraction if possible ---
    if table_rows:
        ri_amount = extract_ri_from_table_rows(table_rows) or extract_ri(raw_text) or ''
    else:
        ri_amount = extract_ri(raw_text) or ''
    not_part_of_capping = extract_not_part_of_capping_from_text(raw_text, pdf_path) or ''
    supervision_charges = extract_supervision_amount_from_table_rows(table_rows) if table_rows else ''
    deposit = extract_sd_amount_from_values(total_dn_amount, ri_amount, not_part_of_capping, supervision_charges) or ''
    gst = extract_gst_sum_from_table_rows(table_rows) if table_rows else ''
    # Demand Note Date extraction
    dn_received_date = extract_demand_note_date_from_text(raw_text) or ''
    # Set DN RECEIVED FROM PARTNER/AUTHORITY- DATE to be the same as Demand Note Date
    dn_received_from_partner_date = dn_received_date
    dn_number = extract_demand_note_reference(raw_text) or ''
    ot_length = extract_section_length(raw_text) or ''
    surface = extract_road_types_from_text(raw_text) or ''
    ground_rent = extract_ground_rent(raw_text, pdf_path) or ''
    administrative_charge = extract_administrative_charge(raw_text) or ''
    chamber_fee = extract_chamber_fee(raw_text) or ''
    hdd_number_of_pits = extract_hdd_number_of_pits_from_text(raw_text) or ''
    surface_wise_length = extract_surface_wise_length(raw_text) or ''
    surface_wise_multiplication_factor = extract_multiplication_factor(raw_text) or ''
    row_application_date = extract_row_application_date_from_text(raw_text) or ''
    difference_days = extract_difference_days(dn_received_date) or ''
    covered_under_capping = extract_covered_under_capping(raw_text) or ''
    hdd_pit_rate = extract_hdd_pit_rate_from_text(raw_text) or ''
    rate_per_meter = extract_rate_per_meter_from_text(raw_text) or ''

    # Calculate non_refundable_cost as total_dn_amount - deposit
    try:
        if total_dn_amount and deposit:
            non_refundable_cost = str(int(float(total_dn_amount) - float(deposit)))
        else:
            non_refundable_cost = ''
    except Exception:
        non_refundable_cost = ''

    # Covered under capping: non_refundable_cost - not_part_of_capping
    try:
        if non_refundable_cost and not_part_of_capping:
            covered_under_capping = str(int(float(non_refundable_cost) - float(not_part_of_capping)))
        else:
            covered_under_capping = ''
    except Exception:
        covered_under_capping = ''

    # UG TYPE extraction
    def extract_ug_type(text):
        ot = False
        hdd = False
        # Try to focus on 'अ)' and 'ब)' sections if present
        import re
        a_section = ''
        b_section = ''
        a_match = re.search(r'अ\)[^\)]*', text)
        b_match = re.search(r'ब\)[^\)]*', text)
        if a_match:
            a_section = a_match.group(0)
        if b_match:
            b_section = b_match.group(0)
        search_texts = [a_section, b_section] if (a_section or b_section) else [text]
        for t in search_texts:
            if 'ओपन ट्रेंच' in t:
                ot = True
            if 'पीट' in t:
                hdd = True
        if ot and hdd:
            return 'OT/HDD'
        elif ot:
            return 'OT'
        elif hdd:
            return 'HDD'
        return ''
    ug_type = extract_ug_type(raw_text)

    # Calculate difference_days as days between today and dn_received_date
    from datetime import datetime
    try:
        if dn_received_date:
            dn_date = datetime.strptime(dn_received_date, "%d/%m/%Y")
            today = datetime.now()
            difference_days = str((today - dn_date).days)
        else:
            difference_days = ''
    except Exception:
        difference_days = ''

    # Compose result
    result = {
        "dn_number": dn_number,
        "dn_received_date": dn_received_date,
        "dn_received_from_partner_date": dn_received_from_partner_date,
        "ot_length": ot_length,
        "dn_length_mtr": ot_length,
        "deposit": deposit,
        "surface": surface,
        "ground_rent": ground_rent,
        "gst": gst,
        "non_refundable_cost": non_refundable_cost,
        "supervision_charges": supervision_charges,
        "reinstallation_amount": ri_amount,
        "administrative_charge": administrative_charge,
        "chamber_fee": chamber_fee,
        "surface_wise_length": surface_wise_length,
        "surface_wise_ri_amount": '',
        "surface_wise_multiplication_factor": surface_wise_multiplication_factor,
        "row_application_date": row_application_date,
        "difference_days": difference_days,
        "total_dn_amount": total_dn_amount,
        "covered_under_capping": covered_under_capping,
        "not_part_of_capping": not_part_of_capping,
        "ri_amount": ri_amount,
        "hdd_pit_rate": hdd_pit_rate,
        "rate_per_meter": rate_per_meter,
        "hdd_number_of_pits": hdd_number_of_pits,
        "ug_type": ug_type,
    }
    return result

# --- Mapping Functions ---
# Mapping from backend extraction keys to frontend preview columns
NMMCBACKEND_TO_PREVIEW = {
    "dn_number": "Demand Note Reference number",
    "dn_received_date": "Demand Note Date",
    "dn_received_from_partner_date": "DN RECEIVED FROM PARTNER/AUTHORITY- DATE",
    "ot_length": "Section Length (Mtr.)",
    "dn_length_mtr": "Section Length (Mtr.)",
    "deposit": "SD Amount",
    "surface": "Road Types - CC/BT/TILES/ Normal Soil/kacha",
    "ground_rent": "Not part of capping (License Fee/Rental Payment /Way Leave charges etc.)",
    "gst": "GST Amount",
    "non_refundable_cost": "Non Refundable Cost (Amount to process for payment should be sum of 'A' + 'B' + 'AA')",
    "supervision_charges": "Supervision Charges",
    "reinstallation_amount": "RI Amount",
    "administrative_charge": "Administrative Charge",
    "chamber_fee": "Chamber Fee",
    "surface_wise_length": "Surface-wise Length",
    "surface_wise_ri_amount": "Surface-wise RI Amount",
    "surface_wise_multiplication_factor": "Surface-wise Multiplication Factor",
    "row_application_date": "ROW APPLICATION  DATE",
    "difference_days": "Difference from, DN date  - DN Sent to Central team (ARTL)",
    "total_dn_amount": "Total DN Amount ( NON REFUNDABLE+SD+ BG+ GST) To be filled by helpdesk team",
    "covered_under_capping": "Covered under capping (Restoration Charges, admin, registration etc.)",
    "not_part_of_capping": "Not part of capping (License Fee/Rental Payment /Way Leave charges etc.)",
    "ri_amount": "RI Amount",
    "hdd_pit_rate": "HDD(PIT RATE)",
    "rate_per_meter": "Rate/mtr- Current DN (UG/OH)",
    "hdd_number_of_pits": "HDD - Number of Pits",
    "ug_type": "UG TYPE( HDD/ OT/ MICROTRENCHING)",
}

SD_BACKEND_TO_PREVIEW = {
    "dn_number": "DN No",
    "dn_received_date": "DN Date",
    "deposit": "SD Amount",
    # All other columns are static or blank
}

# Use the frontend's column lists
PREVIEW_NON_REFUNDABLE_COLUMNS = [
    "Intercity/Intracity- Deployment Intercity/intracity- O&M FTTH- Deployment FTTH-O&M",
    "BUSINESS UNIT",
    "Circle",
    "City",
    "Demand Note Reference number",
    "LM/BB/FTTH",
    "Type (UG/OH)",
    "Capping/Non Capping",
    "UG TYPE( HDD/ OT/ MICROTRENCHING)",
    "Road Types - CC/BT/TILES/ Normal Soil/kacha",
    "HDD - Number of Pits",
    "OH (EB Poles/MC Poles/Own Poles)",
    "NO OF POLES",
    "RAILWAY CROSSING/ PIPELINE CROSSING( No of crossing)",
    "GO RATE",
    "PREVIOUS DN RATE",
    "Rate/mtr- Current DN (UG/OH)",
    "Annual Rate/Pole( current DN)",
    "HDD(PIT RATE)",
    "Section Length (Mtr.)",
    "Total Route (MTR)",
    "RAILWAY/ PIPELINE/ EACH CROSSING RATE",
    "Reason (Current rate is more than GO or Previous DN)",
    "Annual Lease/ rent amount",
    "Renewal Lease/Rent date",
    "Not part of capping (License Fee/Rental Payment /Way Leave charges etc.)",
    "Covered under capping (Restoration Charges, admin, registration etc.)",
    "Non Refundable Cost (Amount to process for payment should be sum of 'A' + 'B' + 'AA')",
    "Cost type with Cost Breakup EG.. - PROCESING FEES/ SUPERVISOIN CHARGE/ ADMIN FEES/ LICENSE FEES etc etc.",
    "GST Amount",
    "BG Amount",
    "SD Amount",
    "ROW APPLICATION  DATE",
    "Demand Note Date",
    "DN RECEIVED FROM PARTNER/AUTHORITY- DATE",
    "Difference from, DN date  - DN Sent to Central team (ARTL)",
    "REASON FOR DELAY (>2 DAYS)",
    "Total DN Amount ( NON REFUNDABLE+SD+ BG+ GST) To be filled by helpdesk team",
    "Supplier Code( if team have) To be filled by helpdesk team",
    "Supplier site name( if team have) To be filled by helpdesk team",
    "Locator Code (material)",
    "Authority( email address)",
    "Authority",
    "BENEFICIERY NAME",
    "Mode of payment(DD/ONLINE-URL/ONLINE-NEFT/BHARATKOSH",
    "EXECUTION PARTNER NAME",
    "Payable (Authority) Location",
    "Printing Location",
    "PO No.",
    "Business NFA NUMBER (Approved CAF) To be filled by helpdesk team",
    "Route Name(As per CWIP)",
    "Section Name for ROW(As per CWIP)",
    "NSG ID(As per CWIP)/CWO NO.",
    "Total Amount as per capping MB(Partner Scope)",
    "Cost type(restoration/ supervison/ agency changes/ admin etc)",
    "Total Amount as per capping MB(Not in Partner Scope)",
    "Cost type (way leave charges/ rent/ license etc)",
    "Permission Type (Primary/ Secondary)",
    "Additional Remarks",
]
PREVIEW_SD_COLUMNS = [
    "SD OU Circle Name", "Execution Partner Vendor Code", "Execution Partner Vendor Name", "Execution Partner GBPA PO No.",
    "GIS Code", "M6 Code", "Locator ID", "Mother Work Order", "Child Work Order", "FA Location", "Partner PO circle",
    "Unique route id", "Supplier Code", "Supplier site name", "NFA no.", "Payment type", "DN No", "DN Date", "SD Amount", "SD Time Period"
]

def map_nmmc_non_refundable_output(fields):
    """
    Map NMMC fields to non-refundable output table format using comprehensive field mapping.
    """
    from constants.comprehensive_field_mapping import map_parser_to_standard, convert_standard_to_table, ensure_all_fields_present
    
    # First convert parser fields to standard format
    standard_fields = map_parser_to_standard(fields, "nmmc")
    
    # Then convert to non-refundable table format
    result = convert_standard_to_table(standard_fields, "non_refundable")
    
    # Ensure all required fields are present
    result = ensure_all_fields_present(result, "non_refundable")
    
    # Add static values
    for col in PREVIEW_NON_REFUNDABLE_COLUMNS:
        if col not in result and col in STATIC_VALUES:
            result[col] = STATIC_VALUES[col]
        elif col not in result:
            result[col] = ''
    
    return result

def map_nmmc_sd_output(fields):
    """
    Map NMMC fields to SD output table format using comprehensive field mapping.
    """
    from constants.comprehensive_field_mapping import map_parser_to_standard, convert_standard_to_table, ensure_all_fields_present
    
    # First convert parser fields to standard format
    standard_fields = map_parser_to_standard(fields, "nmmc")
    
    # Then convert to SD table format
    result = convert_standard_to_table(standard_fields, "sd")
    
    # Ensure all required fields are present
    result = ensure_all_fields_present(result, "sd")
    
    # Add static values for SD output
    result["SD OU Circle Name"] = "TNL-FF-Maharashtra"
    result["Execution Partner Vendor Code"] = "632607"
    result["Execution Partner Vendor Name"] = "Excel Telesonic India Private Limited"
    result["Locator ID"] = "61027-IP01-2948564-CONT1210"
    result["Partner PO circle"] = "Mumbai"
    result["NFA no."] = "1-Business/156/205658"
    result["Payment type"] = "DD"
    result["SD Time Period"] = "2 years"
    
    # Return as headers, row array
    return PREVIEW_SD_COLUMNS, [result.get(col, '') for col in PREVIEW_SD_COLUMNS]

def map_nmmc_validation_table(fields):
    """Map all fields to the validation table format (full superset for DN master upload)."""
    # Use all possible fields for validation
    return fields.copy()

# --- Non-Refundable Output ---
def non_refundable_request_parser(pdf_path, manual_values=None, file_hash=None):
    """
    Extraction logic for NMMC Non Refundable Request Parser.
    Uses robust OCR/text extraction and returns a dict with all fields needed by the frontend validation table.
    """
    # Use the robust extraction function
    extracted = extract_nmmc_all_fields(pdf_path)
    
    # Import the comprehensive field mapping
    from constants.comprehensive_field_mapping import map_parser_to_standard, convert_standard_to_table, ensure_all_fields_present
    
    # Map parser fields to standard field names
    standard_fields = map_parser_to_standard(extracted, "nmmc")
    
    # Convert to validation table format (with display names)
    validation_result = convert_standard_to_table(standard_fields, "validation")
    
    # Ensure all required fields are present
    validation_result = ensure_all_fields_present(validation_result, "validation")
    
    # Import the comprehensive field mapping to get all required fields
    from constants.comprehensive_field_mapping import ALL_NON_REFUNDABLE_FIELDS
    
    # Convert validation_result to non-refundable table format
    non_refundable_result = convert_standard_to_table(standard_fields, "non_refundable")
    non_refundable_result = ensure_all_fields_present(non_refundable_result, "non_refundable")
    
    # Add hardcoded values for NMMC (using display names as keys)
    hardcoded_values = {
        "Intercity/Intracity- Deployment Intercity/intracity- O&M FTTH- Deployment FTTH-O&M": "Intercity/Intracity - Deployment",
        "BUSINESS UNIT": "TNL-FF-Maharashtra",
        "Circle": "MUM",
        "City": "MUM",
        "Capping/Non Capping": "Non capping",
        "Cost type with Cost Breakup EG.. - PROCESING FEES/ SUPERVISOIN CHARGE/ ADMIN FEES/ LICENSE FEES etc etc.": "Restoration Charges",
        "Authority( email address)": "dmc_zone_@nmmc.gov.in",
        "Authority": "Navi Mumbai Municipal Corporation",
        "BENEFICIERY NAME": "Navi Mumbai Municipal Corporation",
        "Mode of payment(DD/ONLINE-URL/ONLINE-NEFT/BHARATKOSH": "DD",
        "EXECUTION PARTNER NAME": "Excel Telesonic India Private Limited",
        "Payable (Authority) Location": "Navi Mumbai",
        "Cost type(restoration/ supervison/ agency changes/ admin etc)": "Restoration Charges",
        "Permission Type (Primary/ Secondary)": "Primary",
        "Type (UG/OH)": "UG",
        "Project Name": "Mumbai Fiber Refresh LMC",
    }
    
    # Calculate "Non Refundable Cost" as sum of covered_under_capping + not_part_of_capping
    covered_under_capping = non_refundable_result.get("Covered under capping (Restoration Charges, admin, registration etc.)", "0")
    not_part_of_capping = non_refundable_result.get("Not part of capping (License Fee/Rental Payment /Way Leave charges etc.)", "0")
    
    try:
        covered_val = float(covered_under_capping) if covered_under_capping else 0
        not_part_val = float(not_part_of_capping) if not_part_of_capping else 0
        non_refundable_cost = covered_val + not_part_val
        hardcoded_values["Non Refundable Cost (Amount to process for payment should be sum of 'A' + 'B' + 'AA')"] = str(non_refundable_cost)
    except (ValueError, TypeError):
        hardcoded_values["Non Refundable Cost (Amount to process for payment should be sum of 'A' + 'B' + 'AA')"] = ""
    
    # Add calculated values for missing fields
    if "Section Length (Mtr.)" not in non_refundable_result or not non_refundable_result["Section Length (Mtr.)"]:
        # Use Total Route (MTR) as Section Length if available
        total_route = non_refundable_result.get("Total Route (MTR)", "")
        if total_route:
            hardcoded_values["Section Length (Mtr.)"] = total_route
    
    # Add extracted date values to hardcoded values
    row_app_date = extracted.get("row_application_date", "")
    if row_app_date:
        hardcoded_values["ROW APPLICATION  DATE"] = row_app_date
    
    demand_note_date = extracted.get("dn_received_date", "")
    if demand_note_date:
        hardcoded_values["Demand Note Date"] = demand_note_date
    
    diff_days = extracted.get("difference_days", "")
    if diff_days:
        hardcoded_values["Difference from, DN date  - DN Sent to Central team (ARTL)"] = diff_days
    
    # Merge hardcoded values into non_refundable_result AFTER ensure_all_fields_present
    non_refundable_result.update(hardcoded_values)
    
    # Print comprehensive field mapping for Non-Refundable table
    print("\n" + "="*80)
    print("NMMC NON-REFUNDABLE TABLE - COMPLETE FIELD MAPPING")
    print("="*80)
    
    # Show every field in the non-refundable table
    for field in ALL_NON_REFUNDABLE_FIELDS:
        value = non_refundable_result.get(field, "")
        if value:
            print(f"✓ {field}: {value}")
        else:
            print(f"✗ {field}: (blank)")
    
    print("="*80 + "\n")
    
    return non_refundable_result

def sd_parser_from_ai_result(ai_result):
    """
    SD Parser for NMMC: outputs a tuple of (headers, row) for SD Excel, using already-extracted AI result dict.
    """
    alt_headers = [
        "SD OU Circle Name", "Execution Partner Vendor Code", "Execution Partner Vendor Name", "Execution Partner GBPA PO No.",
        "GIS Code", "M6 Code", "Locator ID", "Mother Work Order", "Child Work Order", "FA Location", "Partner PO circle",
        "Unique route id", "Supplier Code", "Supplier site name", "NFA no.", "Payment type", "DN No", "DN Date", "SD Amount", "SD Time Period"
    ]
    row = []
    for h in alt_headers:
        if h == "SD OU Circle Name":
            value = "TNL-FF-Maharashtra"
        elif h == "Execution Partner Vendor Code":
            value = "632607"
        elif h == "Execution Partner Vendor Name":
            value = "Excel Telesonic India Private Limited"
        elif h == "Locator ID":
            value = "61027-IP01-2948564-CONT1210"
        elif h == "Partner PO circle":
            value = "Mumbai"
        elif h == "NFA no.":
            value = "1-Business/156/205658"
        elif h == "Payment type":
            value = "DD"
        elif h == "SD Time Period":
            value = "2 years"
        elif h == "DN No":
            value = ai_result.get("dn_number", "")
        elif h == "DN Date":
            value = ai_result.get("dn_received_date", "")
        elif h == "SD Amount":
            value = ai_result.get("deposit", "")
        else:
            value = ""
        row.append(value)
    
    # Print comprehensive field mapping for SD table
    print("\n" + "="*80)
    print("NMMC SD TABLE - COMPLETE FIELD MAPPING")
    print("="*80)
    
    # Import the comprehensive field mapping to get all required fields
    from constants.comprehensive_field_mapping import ALL_SD_FIELDS
    
    # Create the complete SD row data
    sd_data = {}
    for i, header in enumerate(alt_headers):
        sd_data[header] = row[i] if i < len(row) else ""
    
    # Show every field in the SD table
    for field in ALL_SD_FIELDS:
        value = sd_data.get(field, "")
        if value:
            print(f"✓ {field}: {value}")
        else:
            print(f"✗ {field}: (blank)")
    
    print("="*80 + "\n")
    
    return alt_headers, row

# Optionally, a cache or parameter can be used to pass the AI result
# For now, update sd_parser to accept an ai_result param and use it if provided

def sd_parser(pdf_path, manual_values=None, ai_result=None, file_hash=None):
    """
    SD Parser for NMMC: outputs a tuple of (headers, row) for SD Excel.
    If ai_result is provided, use it directly. Otherwise, extract from the PDF (legacy fallback).
    """
    if ai_result is not None:
        return sd_parser_from_ai_result(ai_result)
    # Legacy fallback: extract from PDF (should not be used in new flow)
    raw_text = ''
    if pdf_path:
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            for page_num, pil_img in enumerate(pages):
                temp_img_path = f"_nmmc_tmp_page_{page_num+1}.png"
                pil_img.save(temp_img_path, format="PNG")
                # OCR the image
                img = np.array(pil_img.convert('L'))
                text = pytesseract.image_to_string(img, config='--psm 6', lang='mar+eng')
                raw_text += '\n' + text
        except Exception as e:

            return ([], [])

    import re, json
    if raw_text.strip().startswith('```'):
        raw_text = re.sub(r'^```[a-zA-Z]*', '', raw_text)
        raw_text = re.sub(r'```$', '', raw_text)
        raw_text = raw_text.strip()
    try:
        ai_result = json.loads(raw_text)
    except Exception:
        ai_result = {}
    return sd_parser_from_ai_result(ai_result)

def parse_nmmc_application_table(pdf_path):
    """
    Given a PDF path, extract the NMMC application table robustly.
    Tries both lattice and stream, picks the best table, combines data rows, and prints the result.
    """
    import camelot
    import pandas as pd
    # Try both flavors, pick the table with the most columns
    tables_lattice = camelot.read_pdf(pdf_path, pages="1-end", flavor="lattice")
    tables_stream = camelot.read_pdf(pdf_path, pages="1-end", flavor="stream")
    all_tables = list(tables_lattice) + list(tables_stream)
    if not all_tables:
        return {}
    # Pick the table with the most columns and at least 5 columns
    best_table = max([t for t in all_tables if t.df.shape[1] >= 5], key=lambda t: t.df.shape[1], default=None)
    if best_table is None:
        return {}
    df = best_table.df
    # Drop all-empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    df = df.reset_index(drop=True)
    # Heuristic: header is first 7-9 rows, data is after
    header_rows = df.iloc[:9]
    data_rows = df.iloc[9:15]  # Try to grab 5-6 rows after header
    # Build header by joining non-empty cells in each column
    header = []
    for col in df.columns:
        col_header = []
        for i in range(len(header_rows)):
            val = str(header_rows.iloc[i][col]).strip()
            if val:
                col_header.append(val)
        header.append(' '.join(col_header))
    # Combine data rows into one row per column
    combined = []
    for col in df.columns:
        col_vals = [str(data_rows.iloc[i][col]).strip() for i in range(len(data_rows))]
        col_vals = [v for v in col_vals if v]
        combined.append(' '.join(col_vals))
    # Map header to combined data
    result = {h.replace('\n', ' ').replace('  ', ' ').strip(): v for h, v in zip(header, combined)}
    print("[NMMC APPLICATION TABLE HEADER]", header)
    print("[NMMC APPLICATION TABLE DATA]", combined)
    print("[NMMC APPLICATION TABLE PARSED]", result)
    return result

def extract_all_fields_for_testing(pdf_path):
    """
    Extracts all fields for validation table or field mapping.
    For now, returns a dictionary with all expected keys and placeholder values.
    """
    return {
        "Demand Note Reference number": None,
        "Section Length": None,
        "GST Amount": None,
        "SD Amount": None,
        "ROW APPLICATION  DATE": None,
        "DN Received Date": None,
        "Demand Note Date": None,
        "Difference Days": None,
        "Total DN Amount": None,
        "Road Types": None,
        "Surface-wise RI Amount": None,
        "Covered under capping": None,
        "Not part of capping": None,
        "Non Refundable Cost": None,
        "RI Amount": None,
        "Ground Rent": None,
        "Administrative Charge": None,
        "Supervision Charges": None,
        "Chamber Fee": None,
        "GST (custom)": None,
        "Surface-wise Multiplication Factor": None,
        "surface_wise_length": None,
        # ...add more extra fields here as needed...
    }

def translate_marathi_to_english(text):
    import requests
    try:
        response = requests.post(
            "https://libretranslate.de/translate",
            data={
                "q": text,
                "source": "mr",
                "target": "en",
                "format": "text"
            },
            headers={"accept": "application/json"}
        )
        if response.status_code == 200:
            return response.json().get("translatedText", "")
        else:
            return f"[Translation failed: {response.status_code}] {response.text}"
    except Exception as e:
        return f"[Translation service unavailable: {e}]"

def extract_road_types_from_text(text):
    """
    Extract all road types from OCR text, map to English, allow duplicates, preserve order, and join with '/'.
    This version is robust to OCR noise, repeated types, minor misspellings, and table structure.
    """
    import re
    # List of (regex, English name) for all supported types, including common OCR errors
    road_type_patterns = [
        (r'डांबरी|dambri|bituminous', 'Bituminous (Dambri)'),
        (r'काँक्रीट|concrete|konkrit|konkreet', 'Concrete'),
        (r'टाइल्स|टाईल्स|tiles|taills|tiels', 'Tiles'),
        (r'माती|साधारण माती|normal soil|maati|mati', 'Normal Soil'),
        (r'कच्चा|kachcha|kacha', 'Kacha'),
    ]
    text_lc = text.lower()
    found_types = []
    # 1. Try to extract from table-like lines (with 'प्रकार', 'road', 'type', 'दर/र.मी', or numbers)
    for line in text.splitlines():
        line_lc = line.lower()
        if re.search(r'प्रकार|road|type|दर/र.मी|[0-9]', line_lc):
            for pattern, english in road_type_patterns:
                for match in re.finditer(pattern, line_lc, re.IGNORECASE):
                    found_types.append((match.start(), english))
    if found_types:
        found_types = sorted(found_types, key=lambda x: x[0])
        ordered_types = [t for _, t in found_types]
        return ' / '.join(ordered_types)
    # 2. Fallback: previous logic (all matches in text)
    found_types = []
    for pattern, english in road_type_patterns:
        for match in re.finditer(pattern, text_lc, re.IGNORECASE):
            found_types.append((match.start(), english))
    found_types = sorted(found_types, key=lambda x: x[0])
    ordered_types = [t for _, t in found_types]
    return ' / '.join(ordered_types)

def extract_hdd_number_of_pits_from_text(text):
    """
    Robustly extract the HDD number of pits from OCR text.
    Prioritize extracting the number after 'ब)पीट' or 'पीट -', fallback to first number in a line with 'पीट' or 'खड्डे'.
    """
    import re
    # 1. Look for 'ब[)\.]?\s*पीट\s*[-:]?\s*(\d{1,3})' or 'पीट\s*[-:]?\s*(\d{1,3})'
    match = re.search(r'ब[)\.]?\s*पीट\s*[-:]?\s*(\d{1,3})', text)
    if match:
        return match.group(1)
    match = re.search(r'पीट\s*[-:]?\s*(\d{1,3})', text)
    if match:
        return match.group(1)
    # 2. Look for lines containing 'पीट' or 'खड्डे' and extract the first integer
    for line in text.splitlines():
        if re.search(r'पीट|खड्डे', line):
            match = re.search(r'(\d{1,3})', line)
            if match:
                return match.group(1)
    return ''

def extract_rate_per_meter_from_text(text):
    """
    Extract the OT rate per meter (e.g., 9600) from OCR text.
    Only extract the first number that appears after 'रू' or in the same line as 'दर/र.मी'.
    Return a single value as a string, or empty if not found.
    """
    import re
    for line in text.splitlines():
        # If line contains 'रू', extract the number after it
        if 'रू' in line:
            match = re.search(r'रू\s*(\d{3,6})', line)
            if match:
                return match.group(1)
        # If line contains 'दर/र.मी', extract the number after it or in the same line
        elif 'दर/र.मी' in line:
            found = re.findall(r'(\d{3,6})', line)
            if found:
                return found[0]
    return ''

def extract_hdd_pit_rate_from_text(text):
    """
    Extract the HDD pit rate (e.g., 9600) from the pit section in OCR text.
    Look for lines containing 'पीट' or 'खड्डे', then extract the first 4-6 digit number (rate) from that line or the next line.
    Return as a string, or empty if not found.
    """
    import re
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'पीट|खड्डे', line):
            # Try to find a 4-6 digit number (rate) in the same line
            match = re.search(r'(\d{4,6})', line)
            if match:
                return match.group(1)
            # If not found, try the next line
            if i+1 < len(lines):
                match = re.search(r'(\d{4,6})', lines[i+1])
                if match:
                    return match.group(1)
    return ''

def extract_not_part_of_capping_from_text(text, pdf_path=None):
    """
    Calculate the 'not part of capping' value as 200 times the section length.
    Returns as a string (with two decimals if needed), or '' if section length is missing/invalid.
    """
    section_length = extract_section_length(text)
    try:
        length = float(section_length)
        value = length * 200
        return f"{value:.2f}" if not value.is_integer() else str(int(value))
    except Exception:
        return ''

def extract_total_dn_amount_from_text(text):
    """
    Extract the total DN amount (Non Refundable + SD + BG + GST) from OCR text.
    This is typically found in the "Total DN Amount ( NON REFUNDABLE+SD+ BG+ GST) To be filled by helpdesk team" line,
    or in Marathi lines like 'एकूण रक्‍्कम<9,54,360/-' or 'a+r) = 9,54,360.00'.
    """
    import re
    # 1. Try English pattern
    match = re.search(r'Total DN Amount.*?(\d{5,10}(?:[.,]\d{1,2})?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    # 2. Try Marathi/other patterns
    for line in text.splitlines():
        if re.search(r'एकूण|a\+r|रक्‍्कम', line):
            # Find the first 6-8 digit number with optional commas/decimals
            match = re.search(r'(\d{1,3}(?:,\d{2,3}){1,3}(?:\.\d{1,2})?)', line)
            if match:
                return match.group(1).replace(",", "")
    # 3. Fallback: any large number in the text
    match = re.search(r'(\d{1,3}(?:,\d{2,3}){1,3}(?:\.\d{1,2})?)', text)
    if match:
        return match.group(1).replace(",", "")
    return ''

def extract_ri(text):
    """
    Extract the RI (Reinstallation) amount from OCR text.
    Looks for lines containing 'पर्नस्थापना' (Marathi for Reinstallation) or 'Reinstallation',
    and extracts the first large number (with optional commas/decimals) on the same or next line.
    If not found, fallback to any large number in the text.
    """
    import re
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'पर्नस्थापना|Reinstallation', line, re.IGNORECASE):
            # Try to find a large number in the same line
            match = re.search(r'(\d{1,3}(?:,\d{2,3}){1,3}(?:\.\d{1,2})?)', line)
            if match:
                return match.group(1).replace(",", "")
            # If not found, try the next line
            if i+1 < len(lines):
                match = re.search(r'(\d{1,3}(?:,\d{2,3}){1,3}(?:\.\d{1,2})?)', lines[i+1])
                if match:
                    return match.group(1).replace(",", "")
    # Fallback: any large number in the text
    match = re.search(r'(\d{1,3}(?:,\d{2,3}){1,3}(?:\.\d{1,2})?)', text)
    if match:
        return match.group(1).replace(",", "")
    return ''

def extract_supervision_amount_from_table_rows(rows):
    """
    Robustly extract the Supervision Amount from a table row.
    1. Find any row where any cell contains 'supervision' (case-insensitive, ignore whitespace and punctuation).
    2. From that row, extract all numbers (with or without commas, possibly with trailing/leading non-numeric chars).
    3. Return the largest number found in that row (since sometimes there are multiple numbers, but the largest is usually the amount).
    4. If nothing is found, fallback to searching the entire table for any cell containing 'supervision' and a number.
    """
    import re
    supervision_amounts = []
    for row in rows:
        # Check if any cell in the row contains 'supervision'
        if any(cell and 'supervision' in str(cell).lower() for cell in row):
            for cell in row:
                if cell:
                    # Extract all numbers (with or without commas/decimals)
                    matches = re.findall(r'(\d{2,8}(?:\.\d{1,2})?)', str(cell).replace(',', ''))
                    for m in matches:
                        try:
                            supervision_amounts.append(float(m))
                        except Exception:
                            pass
            if supervision_amounts:
                # Return the largest number found in the row
                return str(int(max(supervision_amounts)))
    # Fallback: search all cells for 'supervision' and a number in the same cell
    for row in rows:
        for cell in row:
            if cell and 'supervision' in str(cell).lower():
                matches = re.findall(r'(\d{2,8}(?:\.\d{1,2})?)', str(cell).replace(',', ''))
                if matches:
                    try:
                        return str(int(max(float(m) for m in matches)))
                    except Exception:
                        continue
    # Fallback: search all cells for a number if the cell contains 'supervision' (even if split)
    for row in rows:
        for cell in row:
            if cell and 'supervision' in str(cell).lower():
                digits = ''.join([c for c in str(cell) if c.isdigit()])
                if digits:
                    return digits
    return ''

def extract_gst_sum_from_table_rows(rows):
    """
    Extracts the sum of GST 'Amt' values from a table row.
    Looks for the 'Total GST' column and the 'Amt' sub-column, or for '18%' in a row,
    and extracts the number in the adjacent cell.
    Returns the sum of all GST amounts found (as string), or '' if not found.
    """
    import re
    gst_amounts = []
    for row in rows:
        for idx, cell in enumerate(row):
            if cell and (('18%' in str(cell)) or ('total gst' in str(cell).lower())):
                # Try to extract the number in the next cell
                if idx + 1 < len(row):
                    next_cell = row[idx + 1]
                    if next_cell:
                        matches = re.findall(r'\d{1,8}(?:\.\d{1,2})?', str(next_cell).replace(',', ''))
                        for m in matches:
                            try:
                                gst_amounts.append(float(m))
                            except Exception:
                                pass
    if gst_amounts:
        return str(int(sum(gst_amounts)))
    return ''

def extract_row_application_date_from_text(text):
    """
    Extracts the ROW Application Date from text.
    Looks for 'संदर्भ' and a date (DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY) after it.
    Returns the date in DD/MM/YYYY format, or '' if not found.
    """
    import re
    # Find the line with 'संदर्भ' and a date
    for line in text.splitlines():
        if 'संदर्भ' in line:
            # Look for a date after 'दिनांक' or in the line
            match = re.search(r'दिनांक\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', line)
            if match:
                date_str = match.group(1)
                # Normalize to DD/MM/YYYY
                date_str = date_str.replace('-', '/').replace('.', '/')
                return date_str
    # Fallback: search for any date after 'संदर्भ' in the text
    match = re.search(r'संदर्भ[^\n]*?([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', text)
    if match:
        date_str = match.group(1)
        date_str = date_str.replace('-', '/').replace('.', '/')
        return date_str
    return ''

def extract_demand_note_date_from_text(text):
    """
    Extracts Demand Note Date from text. Only looks for 'Invoice Date' followed by a date (DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY).
    Returns the date in DD/MM/YYYY format, or '' if not found.
    """
    import re
    # Look for 'Invoice Date' (case-insensitive), optional colon, spaces, then a date
    match = re.search(r'Invoice Date\s*[:\-]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', text, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        date_str = date_str.replace('-', '/').replace('.', '/')
        return date_str
    return ''

def extract_chamber_fee(text):
    """Stub for chamber fee extraction."""
    return ''

def extract_covered_under_capping(text):
    """Stub for covered under capping extraction."""
    return ''

def extract_sd_amount_from_values(total_dn_amount, ri_amount, not_part_of_capping, supervision_charges):
    """
    Calculate SD amount as:
    SD = total_dn_amount - ri_amount - not_part_of_capping - supervision_charges
    All arguments should be strings or floats. Handles missing/empty values gracefully.
    Returns the result as a string (rounded to nearest integer, no commas).
    """
    try:
        tdn = float(str(total_dn_amount).replace(',', '')) if total_dn_amount else 0
        ri = float(str(ri_amount).replace(',', '')) if ri_amount else 0
        npc = float(str(not_part_of_capping).replace(',', '')) if not_part_of_capping else 0
        sup = float(str(supervision_charges).replace(',', '')) if supervision_charges else 0
        sd = tdn - ri - npc - sup
        return str(int(round(sd)))
    except Exception:
        return ''

def extract_demand_note_reference(text):
    """
    Robustly extracts the Demand Note Reference number for NMMC.
    Always returns 'NMMC/Z/' + the extracted part (e.g., 'NMMC/Z/2/266/2025').
    Looks for 'जा.क्र.नमुंमपा/परि' and extracts the following content, cleans it up, and formats as required.
    """
    import re
    for line in text.splitlines():
        if 'जा.क्र.नमुंमपा/परि' in line:
            # Extract everything after 'जा.क्र.नमुंमपा/परि'
            after = line.split('जा.क्र.नमुंमपा/परि')[-1]
            # Remove dashes, pipes, and replace non-digit/non-slash with slash
            after = after.replace('-', '/').replace('॥', '/').replace('।', '/').replace('|', '/').replace(' ', '')
            # Remove any leading/trailing non-digit or slash
            after = re.sub(r'^[^\d/]*', '', after)
            after = re.sub(r'[^\d/]*$', '', after)
            # Replace multiple slashes with single slash
            after = re.sub(r'/+', '/', after)
            # Remove any empty segments
            parts = [p for p in after.split('/') if p]
            final = '/'.join(parts)
            return f'NMMC/Z/{final}'
    # Fallback: previous logic for 'जा.क्र.'
    for line in text.splitlines():
        if 'जा.क्र.' in line:
            after = line.split('जा.क्र.')[-1]
            after = after.replace('-', '/').replace('॥', '/').replace('।', '/').replace('|', '/').replace(' ', '')
            after = re.sub(r'^[^\d/]*', '', after)
            after = re.sub(r'[^\d/]*$', '', after)
            after = re.sub(r'/+', '/', after)
            parts = [p for p in after.split('/') if p]
            final = '/'.join(parts)
            return f'NMMC/Z/{final}'
    # Fallback: any group of numbers and slashes
    match = re.search(r'(\d{1,4}(?:/\d{1,4}){1,3})', text)
    if match:
        return f'NMMC/Z/{match.group(1)}'
    return ''

# --- Test Block ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nmmc.py <path_to_pdf>")
    else:
        pdf_path = sys.argv[1]
        all_fields = extract_nmmc_all_fields(pdf_path)
        print("\n--- Non-Refundable Output ---")
        print(map_nmmc_non_refundable_output(all_fields))
        print("\n--- SD Output ---")
        headers, row = map_nmmc_sd_output(all_fields)
        print(headers)
        print(row)
        print("\n--- Validation Table Output ---")
        print(map_nmmc_validation_table(all_fields))
