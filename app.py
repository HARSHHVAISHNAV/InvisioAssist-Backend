from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import pandas as pd
from rapidfuzz import process, fuzz
import spacy
from google.cloud import vision
from PIL import Image

app = Flask(__name__)
CORS(app)

# ✅ Load Medicine Dataset
CSV_PATH = "sorted_cleaned_dataset.csv"
df = pd.read_csv(CSV_PATH)
df["Name of medicine"] = df["Name of medicine"].str.lower().str.strip()

# ✅ Load Med7 Model for NER-based Recognition
med7 = spacy.load("en_core_med7_lg")

# ✅ Initialize Google Cloud Vision Client
client=vision.ImageAnnotatorClient()

def extract_text_google_vision(image_file):
    """Extract text using Google Cloud Vision OCR."""
    image = vision.Image(content=image_file.read())
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    texts = response.text_annotations
    extracted_text = texts[0].description if texts else ""
    return extracted_text.strip()

def extract_medicine_name_with_med7(text):
    """Extract medicine name using Med7 (NER)."""
    doc = med7(text)
    med_names = [ent.text for ent in doc.ents if ent.label_ == "DRUG"]
    return med_names[0] if med_names else None

@app.route('/')
def home():
    return "Flask server is running with Google Cloud Vision!"

@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']

        # ✅ Extract text using Google Vision OCR
        extracted_text = extract_text_google_vision(image_file)
        print("🔹 Extracted Text:", extracted_text)

        # ✅ Use Med7 for AI-based medicine name recognition
        med7_medicine = extract_medicine_name_with_med7(extracted_text)

        # ✅ If Med7 detects a medicine name, use it; otherwise, use raw OCR text
        search_text = med7_medicine if med7_medicine else extracted_text.lower()

        # ✅ Use Fuzzy Matching if Med7 fails
        match = process.extractOne(search_text, df["Name of medicine"], scorer=fuzz.partial_ratio)
        
        if not match:
            return jsonify({'error': 'No match found', 'extracted_text': extracted_text}), 404

        best_match, score = match[0], match[1]

        if score < 70:
            return jsonify({'error': 'No confident match found', 'extracted_text': extracted_text}), 404

        # ✅ Fetch Medicine Details
        medicine_info = df[df["Name of medicine"] == best_match].iloc[0]
        response = {
            "extracted_text": extracted_text,
            "medicine_name": best_match,
            "full_description": medicine_info["Full Description"],
            "side_effects": medicine_info["Side Effects"]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
