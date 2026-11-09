"""
Sistem Deteksi Website Phishing
Menggunakan model Decision Tree untuk mengklasifikasikan website
sebagai Aman (Legitimate) atau Berbahaya (Phishing).
"""

import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

# Suppress sklearn & joblib serverless warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model_phishing_dt.pkl")
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
        "tooltip": "Apakah website memiliki sertifikat SSL yang valid dan diterbitkan oleh CA terpercaya?",
        "options": [
            ("-1", "Tidak Ada SSL / Sertifikat Tidak Dipercaya"),
            ("0", "SSL Ada, Tapi Berusia Singkat / Mencurigakan"),
            ("1", "SSL Valid, Lengkap & Terpercaya (HTTPS)"),
        ],
    },
    "having_IPhaving_IP_Address": {
        "label": "Alamat IP pada URL",
        "tooltip": "Apakah URL menggunakan alamat IP mentah alih-alih nama domain resmi?",
        "options": [
            ("-1", "Menggunakan Alamat IP Mentah (cth: 192.168.x.x)"),
            ("1", "Menggunakan Nama Domain Standar (cth: domain.com)"),
        ],
    },
    "web_traffic": {
        "label": "Peringkat Web Traffic",
        "tooltip": "Peringkat popularitas website berdasarkan volume kunjungan pengguna global.",
        "options": [
            ("-1", "Traffic Sangat Rendah / Domain Baru Dibuat"),
            ("0", "Traffic Sedang (Rank > 100.000)"),
            ("1", "Traffic Tinggi & Populer (Rank < 100.000)"),
        ],
    },
    "Prefix_Suffix": {
        "label": "Prefix/Suffix pada Domain",
        "tooltip": "Apakah nama domain menggunakan tanda hubung (-) yang lazim dipakai phishing untuk menyamar?",
        "options": [
            ("-1", 'Menggunakan Tanda Hubung "-" (cth: secure-login.com)'),
            ("1", "Tidak Menggunakan Tanda Hubung (Normal)"),
        ],
    },
    "URL_of_Anchor": {
        "label": "Anchor URL (Tautan Jangkar)",
        "tooltip": "Persentase tautan tag <a> yang mengarah ke domain luar atau javascript dummy.",
        "options": [
            ("-1", "Mayoritas Anchor Mengarah ke Domain Luar (> 67%)"),
            ("0", "Sebagian Anchor Mengarah ke Domain Luar (31–67%)"),
            ("1", "Anchor Mengarah ke Domain Sendiri (< 31%)"),
        ],
    },
    "Links_in_tags": {
        "label": "Persentase Link di Tag HTML",
        "tooltip": "Persentase aset di tag <meta>, <script>, dan <link> yang dimuat dari domain luar.",
        "options": [
            ("-1", "Mayoritas Aset Eksternal (> 81%)"),
            ("0", "Sebagian Aset Eksternal (17–81%)"),
            ("1", "Aset Terpusat di Domain Sendiri (< 17%)"),
        ],
    },
}

# Group definitions for the clean 2-column layout (3 features each)
FORM_GROUPS = [
    {
        "title": "Keamanan & Protokol",
        "icon": "🔒",
        "badge_color": "cyan",
        "description": "Evaluasi sertifikat digital SSL, format pengalamatan IP, dan peringkat traffic.",
        "features": ["SSLfinal_State", "having_IPhaving_IP_Address", "web_traffic"],
    },
    {
        "title": "Struktur URL & Analisis Tautan",
        "icon": "🔗",
        "badge_color": "violet",
        "description": "Analisis manipulasi karakter domain, tautan anchor halaman, dan tag script eksternal.",
        "features": ["Prefix_Suffix", "URL_of_Anchor", "Links_in_tags"],
    },
]


def run_prediction(input_data):
    """Build feature vector and run prediction."""
    feature_vector = []
    form_values = {}

    for fname in FEATURE_NAMES:
        if fname in PRIMARY_FEATURES:
            raw_val = input_data.get(fname, DEFAULT_VALUES.get(fname, 1))
            val = int(raw_val)
            form_values[fname] = val
        else:
            val = DEFAULT_VALUES[fname]
        feature_vector.append(val)

    features_array = np.array([feature_vector])

    prediction = int(model.predict(features_array)[0])
    probabilities = model.predict_proba(features_array)[0]

    classes_list = list(model.classes_)
    class_index = classes_list.index(prediction)
    confidence = float(probabilities[class_index]) * 100

    is_safe = prediction == 1
    safe_idx = classes_list.index(1)
    phish_idx = classes_list.index(-1)

    feature_breakdown = []
    for fname in PRIMARY_FEATURES:
        val = form_values[fname]
        meta = FEATURE_META[fname]
        if val == 1:
            status_text = "Normal / Terpercaya"
            status_type = "safe"
        elif val == 0:
            status_text = "Perlu Perhatian"
            status_type = "warning"
        else:
            status_text = "Indikasi Phishing"
            status_type = "danger"

        feature_breakdown.append({
            "name": meta["label"],
            "value": val,
            "status": status_text,
            "type": status_type,
        })

    return {
        "is_safe": is_safe,
        "label": "AMAN — Website Legitimate" if is_safe else "BERBAHAYA — Terdeteksi Phishing",
        "confidence": round(confidence, 1),
        "prediction": prediction,
        "prob_safe": round(float(probabilities[safe_idx]) * 100, 1),
        "prob_phishing": round(float(probabilities[phish_idx]) * 100, 1),
        "form_values": form_values,
        "breakdown": feature_breakdown,
    }


# Route handlers: support '/', '/api', and Vercel's rewrite target '/api/index'
@app.route("/", methods=["GET", "POST"])
@app.route("/api/index", methods=["GET", "POST"])
@app.route("/api", methods=["GET", "POST"])
def index():
    """Handle the main page: show form and (optionally) prediction results."""
    result = None

    if request.method == "POST":
        result = run_prediction(request.form)

    return render_template(
        "index.html",
        result=result,
        groups=FORM_GROUPS,
        meta=FEATURE_META,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic threat detection."""
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    res = run_prediction(data)
    return jsonify({
        "status": "success",
        "prediction": res["prediction"],
        "classification": "Legitimate" if res["is_safe"] else "Phishing",
        "is_safe": res["is_safe"],
        "confidence_percentage": res["confidence"],
        "probabilities": {
            "safe": res["prob_safe"],
            "phishing": res["prob_phishing"],
        },
        "breakdown": res["breakdown"],
    })


@app.route("/favicon.ico")
@app.route("/favicon.png")
def favicon():
    """Prevent 404 logs for browser favicon requests."""
    return "", 204


if __name__ == "__main__":
    app.run(debug=True, port=5000)
