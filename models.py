# models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# ===============================
# LOAD DATASET
# ===============================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

data = pd.read_csv(url, names=columns)

# ===============================
# FITUR & LABEL
# ===============================
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STANDARDISASI
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ===============================
# TRAIN SVM
# ===============================
svm_model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

# ===============================
# SIMPAN MODEL
# ===============================
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(svm_model, open("svm_model.pkl", "wb"))

print("SVM model saved successfully!")
