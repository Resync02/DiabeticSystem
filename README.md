# Diabetes Prediction Web Application using Machine Learning

A web-based machine learning application built with **Flask** to predict whether a person is likely to have diabetes based on several health indicators. The application uses a pre-trained **Support Vector Machine (SVM)** model and displays the prediction along with its confidence probability.

---

## Features

- Web-based user interface using Flask
- Predicts diabetes status from user health data
- Uses a trained Support Vector Machine (SVM) model
- Data preprocessing with StandardScaler
- Displays prediction result with probability score
- Clean and responsive interface

---

## Technologies Used

- Python 3
- Flask
- NumPy
- Scikit-learn
- HTML5
- CSS3
- Pickle

---

## Project Structure

```
MachineL-UAS/
│
├── app.py                  # Main Flask application
├── models.py               # Machine learning utilities
├── scaler.pkl              # StandardScaler model
├── svm_model.pkl           # Trained SVM model
├── knn_model.pkl           # K-Nearest Neighbors model
├── logreg_model.pkl        # Logistic Regression model
├── requirements.txt        # Project dependencies
│
├── templates/
│   ├── index.html          # Input form
│   └── result.html         # Prediction result page
│
├── static/
│   ├── style.css           # Stylesheet
│   └── bg.jpg              # Background image
│
└── README.md
```

---

## Input Features

The prediction is based on the following health parameters:

| Feature | Description |
|----------|-------------|
| Glucose | Blood glucose concentration |
| Blood Pressure | Diastolic blood pressure |
| BMI | Body Mass Index |
| Insulin | 2-Hour serum insulin |
| Age | Patient age |
| Diabetes Pedigree Function | Genetic risk factor |
| Waist | Waist circumference |
| Activity | Physical activity level |

---

## Machine Learning Workflow

1. User enters health information.
2. Input data is standardized using the trained scaler.
3. The processed data is passed to the Support Vector Machine (SVM) model.
4. The model predicts whether the patient is:
   - **Diabetic**
   - **Non-Diabetic**
5. The application displays the prediction and confidence probability.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Resync02/MachineL-UAS.git
```

Move into the project folder:

```bash
cd MachineL-UAS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Example Prediction

### Input

- Glucose: 150
- Blood Pressure: 80
- BMI: 32
- Insulin: 130
- Age: 45
- Diabetes Pedigree Function: 0.62
- Waist: 95
- Activity: 2

### Output

```
Algorithm:
Support Vector Machine (SVM)

Prediction:
Diabetic

Probability:
91.34%
```

---

## Machine Learning Model

The project includes several trained machine learning models:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression

Currently, the web application uses the **Support Vector Machine (SVM)** model for prediction because it provides the best performance for this implementation.

---

## Future Improvements

- Compare predictions from multiple algorithms
- Upload CSV dataset for batch prediction
- User authentication
- Prediction history
- Interactive dashboard with charts
- Deploy to Vercel, Railway, or Render
- REST API implementation

---

## Learning Objectives

This project was developed to:

- Learn Machine Learning implementation using Python
- Understand healthcare prediction models
- Apply Flask in web development
- Integrate trained ML models into web applications
- Practice model deployment for real-world scenarios

---

## Author

**Iqbal Hafidz Ramadhan**

GitHub: https://github.com/Resync02

---

## License

This project is intended for educational and academic purposes. You are welcome to use and modify it for learning and research.
