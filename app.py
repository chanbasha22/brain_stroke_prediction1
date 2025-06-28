from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
model = pickle.load(open('stroke_model.pkl', 'rb'))

# Email sender credentials (replace these)
EMAIL_ADDRESS = 'shaikchan618@gmail.com'
EMAIL_PASSWORD = 'qkbq nrtn qedq pyaq'

def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():
    email = request.form['email']
    return render_template('index.html', user_email=email)

@app.route('/predict', methods=['POST'])
def predict():
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
    email = request.form['email']

    gender = 1 if gender == 'Male' else 0
    ever_married = 1 if ever_married == 'Yes' else 0
    Residence_type = 1 if Residence_type == 'Urban' else 0

    smoking_dict = {'never smoked': 1, 'formerly smoked': 0, 'smokes': 2, 'Unknown': 3}
    smoking_status = smoking_dict.get(smoking_status, 3)

    work_dict = {'Govt_job': 0, 'children': 1, 'Private': 2, 'Self-employed': 3, 'Never_worked': 4}
    work_type = work_dict.get(work_type, 2)

    features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                          work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

    proba = model.predict_proba(features)[0][1]
    threshold = 0.3
    prediction = 1 if proba > threshold else 0
    result = 'Yes' if prediction == 1 else 'No'

    subject = "🧠 Brain Stroke Prediction Result"
    body = f"Hello,\n\nYour stroke prediction result is: {result}\nProbability: {round(proba * 100, 2)}%\n\nStay healthy!"
    send_email(email, subject, body)

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
    email = request.form['email']

    subject = "🩺 Appointment Confirmation"
    body = f"Dear {name},\n\nYour appointment is confirmed.\n\nDetails:\nDate: {date}\nTime: {time}\nDoctor: {doctor}\n\nThank you for using Brain Stroke Predictor."
    send_email(email, subject, body)

    return render_template('appointment_success.html',
                           name=name, age=age, mobile=mobile,
                           date=date, time=time, doctor=doctor)

if __name__ == '__main__':
    app.run(debug=True)
