from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('wine_quality_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature names
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', features=feature_names, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[feature]) for feature in feature_names]
        input_data = np.array([values])
        prediction = model.predict(input_data)[0]
        return render_template('index.html', features=feature_names, prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
