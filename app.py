"""
Sistem Deteksi Website Phishing
Menggunakan model Decision Tree untuk mengklasifikasikan website
sebagai Aman (Legitimate) atau Berbahaya (Phishing).
"""

import os
import warnings

import joblib
import numpy as np
from flask import Flask, render_template, request

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_phishing_dt.pkl")
model = joblib.load(MODEL_PATH)

# Ordered feature names exactly as expected by the model (30 features)
FEATURE_NAMES = list(model.feature_names_in_)

# The 6 primary features exposed in the form
PRIMARY_FEATURES = [
    "SSLfinal_State",
    "URL_of_Anchor",
    "Prefix_Suffix",
    "web_traffic",
    "Links_in_tags",
    "having_IPhaving_IP_Address",  # actual column name in the model
]

# Default values for the remaining 24 features (legitimate‑leaning = 1)
DEFAULT_VALUES = {name: 1 for name in FEATURE_NAMES if name not in PRIMARY_FEATURES}

# Metadata for form rendering: label, tooltip, and select options
FEATURE_META = {
    "SSLfinal_State": {
        "label": "Status SSL / HTTPS",
        "tooltip": "Apakah website memiliki sertifikat SSL yang valid dan terpercaya?",
        "options": [
            ("-1", "Tidak Ada SSL / Tidak Dipercaya"),
            ("0", "SSL Ada, Tapi Mencurigakan"),
            ("1", "SSL Valid & Terpercaya"),
        ],
    },
    "URL_of_Anchor": {
        "label": "Anchor URL (Tautan Jangkar)",
        "tooltip": "Persentase tautan <a> yang mengarah ke domain berbeda dari halaman itu sendiri.",
        "options": [
            ("-1", "Mayoritas Anchor Mengarah ke Domain Lain (> 67%)"),
            ("0", "Sebagian Anchor Mengarah ke Domain Lain (31–67%)"),
            ("1", "Anchor Mengarah ke Domain Sendiri (< 31%)"),
        ],
    },
    "Prefix_Suffix": {
        "label": "Prefix/Suffix pada Domain",
        "tooltip": "Apakah domain URL menggunakan tanda hubung (-) yang sering dipakai situs phishing?",
        "options": [
            ("-1", 'Menggunakan Tanda Hubung "-" (Mencurigakan)'),
            ("1", "Tidak Menggunakan Tanda Hubung (Normal)"),
        ],
    },
    "web_traffic": {
        "label": "Peringkat Web Traffic",
        "tooltip": "Peringkat popularitas website berdasarkan Alexa/Similar Web.",
        "options": [
            ("-1", "Tidak Ada Data Traffic / Sangat Rendah"),
            ("0", "Traffic Sedang (Rank > 100.000)"),
            ("1", "Traffic Tinggi (Rank < 100.000)"),
        ],
    },
    "Links_in_tags": {
        "label": "Persentase Link di Tag HTML",
        "tooltip": "Persentase URL di tag <meta>, <script>, dan <link> yang mengarah ke domain lain.",
        "options": [
            ("-1", "Mayoritas Resource dari Domain Lain (> 81%)"),
            ("0", "Sebagian Resource dari Domain Lain (17–81%)"),
            ("1", "Resource Utama dari Domain Sendiri (< 17%)"),
        ],
    },
    "having_IPhaving_IP_Address": {
        "label": "Alamat IP pada URL",
        "tooltip": "Apakah URL menggunakan alamat IP langsung alih‑alih nama domain?",
        "options": [
            ("-1", "URL Menggunakan Alamat IP (Mencurigakan)"),
            ("1", "URL Menggunakan Nama Domain (Normal)"),
        ],
    },
}

# Group definitions for the form layout
FORM_GROUPS = [
    {
        "title": "Keamanan & Protokol",
        "icon": "🔒",
        "description": "Parameter keamanan koneksi dan sertifikat digital website.",
        "features": ["SSLfinal_State", "having_IPhaving_IP_Address"],
    },
    {
        "title": "Struktur URL & Domain",
        "icon": "🔗",
        "description": "Analisis pola dan struktur URL untuk mendeteksi anomali.",
        "features": ["Prefix_Suffix", "URL_of_Anchor"],
    },
    {
        "title": "Analisis Tautan & Traffic",
        "icon": "📊",
        "description": "Evaluasi sumber daya eksternal dan reputasi traffic website.",
        "features": ["Links_in_tags", "web_traffic"],
    },
]


@app.route("/", methods=["GET", "POST"])
def index():
    """Handle the main page: show form and (optionally) prediction results."""
    result = None

    if request.method == "POST":
        # Build the full feature vector in the correct order
        feature_vector = []
        form_values = {}

        for fname in FEATURE_NAMES:
            if fname in PRIMARY_FEATURES:
                val = int(request.form.get(fname, DEFAULT_VALUES.get(fname, 1)))
                form_values[fname] = val
            else:
                val = DEFAULT_VALUES[fname]
            feature_vector.append(val)

        features_array = np.array([feature_vector])

        # Predict class and probability
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]

        # Classes: -1 = Phishing, 1 = Legitimate
        # model.classes_ is [-1, 1]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        is_safe = int(prediction) == 1

        result = {
            "is_safe": is_safe,
            "label": "AMAN — Website Legitimate" if is_safe else "BERBAHAYA — Terdeteksi Phishing",
            "confidence": round(confidence, 1),
            "prediction": int(prediction),
            "prob_safe": round(float(probabilities[list(model.classes_).index(1)]) * 100, 1),
            "prob_phishing": round(float(probabilities[list(model.classes_).index(-1)]) * 100, 1),
            "form_values": form_values,
        }

    return render_template(
        "index.html",
        result=result,
        groups=FORM_GROUPS,
        meta=FEATURE_META,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

