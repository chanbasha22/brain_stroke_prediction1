# stroke_prediction_app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('stroke_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Encoding categorical variables
        gender = 1 if gender == 'Male' else 0
        ever_married = 1 if ever_married == 'Yes' else 0
        Residence_type = 1 if Residence_type == 'Urban' else 0

        smoking_dict = {
            'never smoked': 1,
            'formerly smoked': 0,
            'smokes': 2,
            'Unknown': 3
        }
        smoking_status = smoking_dict.get(smoking_status, 3)

        work_dict = {
            'Govt_job': 0,
            'children': 1,
            'Private': 2,
            'Self-employed': 3,
            'Never_worked': 4
        }
        work_type = work_dict.get(work_type, 2)

        features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                              work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

        # Predict probability
        proba = model.predict_proba(features)[0][1]
        threshold = 0.3
        prediction = 1 if proba > threshold else 0
        result = 'Yes' if prediction == 1 else 'No'

        return render_template('result.html', prediction=result, probability=round(proba * 100, 2))

@app.route('/book_appointment')
def book_appointment():
    return render_template('book_appointment.html')

@app.route('/submit_appointment', methods=['POST'])
def submit_appointment():
    name = request.form['name']
    age = request.form['age']
    mobile = request.form['mobile']
    date = request.form['date']
    time = request.form['time']
    doctor = request.form['doctor']

    return render_template('appointment_success.html',
                           name=name, age=age, mobile=mobile,
                           date=date, time=time, doctor=doctor)

if __name__ == '__main__':
    app.run(debug=True)