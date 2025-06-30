from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import traceback

# === Initialize Flask ===
app = Flask(__name__)

# === Dataset Loader ===
class ImageDatasetLoader:
    def __init__(self, dataset_dir, image_size=(32, 32)):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.class_names = ['dog', 'cat', 'bird']  # You can also auto-detect if needed

    def load_images(self):
        X, y = [], []
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                try:
                    img = Image.open(filepath).convert('RGB')
                    img_resized = img.resize(self.image_size)
                    X.append(np.array(img_resized).flatten())
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {filepath}: {e}")
        return np.array(X), np.array(y), self.class_names

# === Global Models and Data ===
nb_model = rf_model = mlp_model = scaler = None
class_names = []
X_test = y_test = X_test_scaled = None

# === Train Models ===
def train_all_models():
    global nb_model, rf_model, mlp_model, scaler, class_names, X_test, y_test, X_test_scaled

    loader = ImageDatasetLoader('dataset')
    X, y, class_names = loader.load_images()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model = GaussianNB().fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42).fit(X_train, y_train)
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42).fit(X_train_scaled, y_train)

    print("✅ Models trained")

# === Home Route ===
@app.route("/")
def home():
    return render_template("index.html")

# === Predict Route ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form['model']
        file = request.files['image']
        image = Image.open(file).convert('RGB').resize((32, 32))
        input_vector = np.array(image).flatten().reshape(1, -1)

        if model_name == 'Naive Bayes':
            pred = nb_model.predict(input_vector)
        elif model_name == 'Random Forest':
            pred = rf_model.predict(input_vector)
        elif model_name == 'MLP':
            input_vector_scaled = scaler.transform(input_vector)
            pred = mlp_model.predict(input_vector_scaled)
        else:
            return jsonify({'error': 'Unknown model name'}), 400

        return jsonify({'prediction': class_names[pred[0]]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# === Metrics Route ===
@app.route("/metrics")
def metrics():
    def evaluate(model, X_eval, y_eval):
        y_pred = model.predict(X_eval)
        return {
            "accuracy": round(accuracy_score(y_eval, y_pred) * 100, 1),
            "precision": round(precision_score(y_eval, y_pred, average="macro") * 100, 1),
            "recall": round(recall_score(y_eval, y_pred, average="macro") * 100, 1),
            "f1": round(f1_score(y_eval, y_pred, average="macro") * 100, 1),
            "confusion": confusion_matrix(y_eval, y_pred).tolist()
        }

    return jsonify({
        "Naive Bayes": evaluate(nb_model, X_test, y_test),
        "Random Forest": evaluate(rf_model, X_test, y_test),
        "MLP": evaluate(mlp_model, X_test_scaled, y_test)
    })

# === Run App ===
if __name__ == "__main__":
    train_all_models()
    app.run(debug=True)
