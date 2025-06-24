import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv('brain_stroke.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical values
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Features and target
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanced model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluation
print(df['stroke'].value_counts())
print(classification_report(y_test, model.predict(X_test)))

# Save model
with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(model, f)
