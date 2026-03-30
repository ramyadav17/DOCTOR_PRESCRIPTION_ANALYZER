import re
import pytesseract
from PIL import Image
import joblib
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Load ML assets
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("prescription_model.pkl")

# Load medicine dictionary
medicine_dict = pd.read_csv("medicine_dict.csv")
medicine_list = medicine_dict["drug"].str.lower().tolist()

# Abbreviation expansion
abbrev_map = {
    r"\bOD\b": "once daily",
    r"\bBD\b": "twice daily",
    r"\bTDS\b": "three times daily",
    r"\bSOS\b": "when needed",
    r"\bHS\b": "at bedtime"
}

# ---------- OCR ----------
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


# ---------- CLEANING ----------
def clean_text(text):
    text = str(text).lower()

    for abbr, full in abbrev_map.items():
        text = re.sub(abbr.lower(), full, text)

    # keep line breaks for line parsing
    text = re.sub(r"[^a-z0-9\s\n]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------- NEW CORE LOGIC ----------
def parse_prescription_lines(text, medicine_list):

    lines = re.split(r"\n", text)

    results = []

    dosage_pattern = r"\b\d+\s?(?:mg|ml|g)\b"
    frequency_pattern = r"\b(once daily|twice daily|three times daily|when needed)\b"
    timing_pattern = r"\b(after meals?|before meals?|at night|at bedtime|before food)\b"
    duration_pattern = r"\b\d+\s?days?\b"

    for line in lines:
        line = line.strip().lower()
        if not line:
            continue

        # Find all medicine positions
        matches = []
        for med in medicine_list:
            for m in re.finditer(r"\b" + re.escape(med) + r"\b", line):
                matches.append((m.start(), med))

        # Sort by position
        matches.sort()

        if not matches:
            continue

        # Split line into chunks per medicine
        for i in range(len(matches)):
            start_pos = matches[i][0]
            med = matches[i][1]

            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(line)

            chunk = line[start_pos:end_pos]

            result = {
                "Medicine": med,
                "Dosage": ", ".join(set(re.findall(dosage_pattern, chunk))),
                "Frequency": ", ".join(set(re.findall(frequency_pattern, chunk))),
                "Timing": ", ".join(set(re.findall(timing_pattern, chunk))),
                "Duration": ", ".join(set(re.findall(duration_pattern, chunk))),
            }

            results.append(result)

    return results


# ---------- MAIN STRUCTURED FUNCTION ----------
def extract_structured_data(text):
    cleaned = clean_text(text)
    structured = parse_prescription_lines(cleaned, medicine_list)
    return structured


def split_into_sentences(text):
    keywords = ["take", "tab", "cap", "apply", "avoid", "review", "follow"]
    words = text.split()
    lines = []
    current = []

    for word in words:
        if word in keywords and current:
            lines.append(" ".join(current))
            current = []

        current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines

def clean_noise(text):
    text = text.lower()
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", text)
    text = re.sub(r"\b\d{2}\s\d{2}\s\d{4}\b", "", text)
    text = re.sub(r"\b(rx|rk)\b", "", text)
    return text

# ---------- CLASSIFICATION ----------
def classify_lines(text):
    cleaned = clean_text(text)
    cleaned = clean_noise(cleaned)
    lines = split_into_sentences(cleaned)
    results = []

    for line in lines:
        vec = vectorizer.transform([line])

        label = model.predict(vec)[0]

        results.append({
            "Instruction": line,
            "Type": label
        })

    return results