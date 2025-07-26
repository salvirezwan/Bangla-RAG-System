import os
import sys
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set the TESSDATA_PREFIX environment variable
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def extract_text_from_pdf(pdf_path, output_txt_path="HSC26_Bangla_1st_Paper_ocr.txt", lang="ben+eng", dpi=300):
    print("[...] Starting OCR extraction from PDF using Tesseract...")

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"[!] Error loading PDF: {e}")
        return

    with open(output_txt_path, "w", encoding="utf-8") as output_file:
        for i, page in enumerate(tqdm(images, desc="Processing pages")):
            try:
                text = pytesseract.image_to_string(page, lang=lang)
                output_file.write(f"\n\n--- Page {i + 1} ---\n\n{text}\n")
            except Exception as e:
                print(f"[!] Error processing page {i + 1}: {e}")

    print(f"[✔] OCR extraction complete. Output saved to: {output_txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <PDF_FILE_PATH>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    extract_text_from_pdf(pdf_path)







