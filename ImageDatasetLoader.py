from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import traceback


metrics = {}  # holds evaluation results for each model

# === Dataset Loader ===
class ImageDatasetLoader:
    def __init__(self, dataset_dir, image_size=(32, 32)):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.class_names = ['dog', 'cat', 'bird']  # Adjust based on your dataset folders

    def load_images(self):
        X, y = [], []
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️ Warning: {class_dir} does not exist!")
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

# === Metric Printer ===
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    cm = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON

    metrics[name] = {
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'confusion_matrix': cm
    }

    print(f"\n📊 {name} Results:")
    print(f"Accuracy:  {acc:.1f}%")
    print(f"Precision: {prec:.1f}%")
    print(f"Recall:    {rec:.1f}%")
    print(f"F1-Score:  {f1:.1f}%")
    print("Confusion Matrix:")
    print(cm)


# === Flask App Setup ===
app = Flask(__name__)

nb_model = None
dt_model = None
mlp_model = None
scaler = None
class_names = []

# === Training Routine ===
def train_all_models():
    global nb_model, dt_model, mlp_model, scaler, class_names

    loader = ImageDatasetLoader('dataset')
    X, y, class_names = loader.load_images()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Naive Bayes
    print("\n=== Training Naive Bayes ===")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print_metrics("Naive Bayes", y_test, y_pred_nb)

    # Decision Tree
    print("\n=== Training Decision Tree ===")
    dt_model = DecisionTreeClassifier(max_depth=15, random_state=42, min_samples_split=5, min_samples_leaf=2)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    print_metrics("Decision Tree", y_test, y_pred_dt)

    # Neural Network
    print("\n=== Training Neural Network (MLP) ===")
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    print_metrics("Neural Network (MLP)", y_test, y_pred_mlp)

    print("\n✅ All models trained and evaluated.")

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form['model']
        file = request.files['image']
        image = Image.open(file).convert('RGB').resize((32, 32))
        input_vector = np.array(image).flatten().reshape(1, -1)

        if model_name == 'Naive Bayes':
            pred = nb_model.predict(input_vector)
        elif model_name == 'Decision Tree':
            pred = dt_model.predict(input_vector)
        elif model_name == 'MLP':
            input_vector_scaled = scaler.transform(input_vector)
            pred = mlp_model.predict(input_vector_scaled)
        else:
            return jsonify({'error': 'Unknown model name'}), 400

        return jsonify({'prediction': class_names[pred[0]]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/sample/<category>/<filename>')
def serve_image(category, filename):
    return send_from_directory(f'dataset/{category}', filename)


@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)


# === Main Run ===
if __name__ == '__main__':
    train_all_models()
    app.run(debug=True)