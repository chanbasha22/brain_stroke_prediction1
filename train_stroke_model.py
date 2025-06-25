import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

# Load and clean dataset
df = pd.read_csv('brain_stroke.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical features
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['ever_married'] = LabelEncoder().fit_transform(df['ever_married'])
df['work_type'] = LabelEncoder().fit_transform(df['work_type'])
df['Residence_type'] = LabelEncoder().fit_transform(df['Residence_type'])
df['smoking_status'] = LabelEncoder().fit_transform(df['smoking_status'])

# Split into features and label
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_sm))

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sm, y_train_sm)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(model, f)
