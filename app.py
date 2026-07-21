from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load scaler & SVM model
scaler = pickle.load(open('scaler.pkl', 'rb'))
svm = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form
    input_data = [
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['BMI']),
        float(request.form['Insulin']),
        float(request.form['Age']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Waist']),
        float(request.form['Activity'])
    ]

    # Standarisasi input
    input_scaled = scaler.transform([input_data])

    # Prediksi menggunakan SVM
    probability = svm.predict_proba(input_scaled)[0][1]
    prediction = "Diabetic" if probability >= 0.5 else "Non-Diabetic"

    results = {
        'Algorithm': 'Support Vector Machine (SVM)',
        'Prediction': prediction,
        'Probability': round(probability * 100, 2)
    }

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
