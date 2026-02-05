from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import torch
from transformers import AutoTokenizer
from model import HybridBERT  # make sure you created model.py
from Credentials import register_user, login_user
# ------------------------------
# Initialize Flask
# ------------------------------
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS so React frontend can call Flask

# ------------------------------
# Load model + tokenizer
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = HybridBERT()
model.load_state_dict(torch.load("hybridbert_bot_detector_v21.pth", map_location=device))
model.to(device)
model.eval()

# ------------------------------
# Prediction function
# ------------------------------
def predict_user(text, metadata):
    # Encode text with BERT tokenizer
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Convert metadata safely
    metadata_tensor = torch.tensor([[
        metadata.get("followers_count", 0),
        metadata.get("friends_count", 0),
        metadata.get("listed_count", 0),
        metadata.get("statuses_count", 0),
        int(metadata.get("verified", 0))  # boolean -> int
    ]], dtype=torch.float32).to(device)

    # Run through model
    with torch.no_grad():
        logits = model(input_ids, attention_mask, metadata_tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

    return {
        "prediction": "Bot 🤖" if pred == 1 else "Human 🧑",
        "confidence": round(prob, 4)
    }

# ------------------------------
# API Route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        metadata = data.get("metadata", {})

        if not text:
            return jsonify({"error": "Text input is required"}), 400

        result = predict_user(text, metadata)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    msg = register_user(username, password)
    return jsonify({"message": msg}), 200

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    success, msg = login_user(username, password)
    if success:
        return jsonify({"message": msg}), 200
    else:
        return jsonify({"error": msg}), 400

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
