import os
import pickle
import numpy as np
import torch
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# Load model and embeddings
model, preprocess = None, None
file_names, embeddings = None, None


def load_model_and_embeddings():
    global model, preprocess, file_names, embeddings
    model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
    with open("image_embeddings.pickle", "rb") as f:
        embeddings_data = pickle.load(f)
    file_names = embeddings_data["file_name"]
    embeddings = np.array(embeddings_data["embedding"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/find_similar_images_text", methods=["POST"])
def find_similar_images_text():
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    with torch.no_grad():
        # Tokenize and encode text
        text_tokens = tokenizer.tokenize([text]).to('cpu')
        text_embedding = F.normalize(model.encode_text(text_tokens)).squeeze(0).numpy()

    # Calculate cosine similarities
    similarities = [
        np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb))
        for emb in embeddings
    ]

    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_results = [
        {"file": file_names[i], "similarity": float(similarities[i])} for i in top_indices
    ]

    return jsonify(top_results)


@app.route("/find_similar_images_image", methods=["POST"])
def find_similar_images_image():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    with torch.no_grad():
        # Preprocess and encode image
        image = preprocess(Image.open(image_file)).unsqueeze(0)
        image_embedding = F.normalize(model.encode_image(image)).squeeze(0).numpy()

    # Calculate cosine similarities
    similarities = [
        np.dot(image_embedding, emb) / (np.linalg.norm(image_embedding) * np.linalg.norm(emb))
        for emb in embeddings
    ]

    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_results = [
        {"file": file_names[i], "similarity": float(similarities[i])} for i in top_indices
    ]

    return jsonify(top_results)


@app.route("/find_combined_similarity", methods=["POST"])
def find_combined_similarity():
    text = request.form.get("text")
    image_file = request.files.get("image")
    weight = float(request.form.get("weight", 0.5))  # Default to 0.5 if not provided

    if not text or not image_file:
        return jsonify({"error": "Both text and image are required"}), 400

    with torch.no_grad():
        # Tokenize and encode text
        text_tokens = tokenizer.tokenize([text]).to('cpu')
        text_embedding = F.normalize(model.encode_text(text_tokens)).squeeze(0).numpy()

        # Preprocess and encode image
        image = preprocess(Image.open(image_file)).unsqueeze(0)
        image_embedding = F.normalize(model.encode_image(image)).squeeze(0).numpy()

    # Combine text and image similarities based on weight
    similarities = [
        weight * np.dot(image_embedding, emb) / (np.linalg.norm(image_embedding) * np.linalg.norm(emb))
        + (1 - weight) * np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb))
        for emb in embeddings
    ]

    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_results = [
        {"file": file_names[i], "similarity": float(similarities[i])} for i in top_indices
    ]

    return jsonify(top_results)


@app.route("/image/<filename>")
def serve_image(filename):
    image_path = os.path.join("coco_images_resized", filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return jsonify({"error": f"Image {filename} not found"}), 404


if __name__ == "__main__":
    load_model_and_embeddings()
    app.run(debug=True)
