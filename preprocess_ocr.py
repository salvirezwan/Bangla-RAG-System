import re

def clean_bangla_text(text):
    """
    Cleans Bangla OCR text by removing:
    - English letters and digits (but keeps Bangla digits)
    - Special characters
    - Repeated punctuation
    - Extra whitespace
    """
    # Remove English letters and digits (but not Bangla digits)
    text = re.sub(r'[a-zA-Z0-9]', '', text)

    # Remove special characters and non-Bangla punctuations
    text = re.sub(r'[~`@#$%^&*()_+=\[\]{}<>|\\/:;"\',?]', '', text)

    # Replace multiple Bangla/English punctuation marks with single Bangla danda
    text = re.sub(r'[।.!?]{2,}', '।', text)

    # Normalize whitespace (multiple spaces or newlines)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    input_path = "HSC26_Bangla_1st_Paper_ocr.txt"
    output_path = "HSC26_Bangla_1st_Paper_cleaned.txt"

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        print(f"✅ Loaded {input_path}")

        cleaned_text = clean_bangla_text(raw_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"✅ Cleaned text saved to {output_path}")

    except FileNotFoundError:
        print(f"❌ File '{input_path}' not found.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
