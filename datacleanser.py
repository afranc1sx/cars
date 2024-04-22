import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from joblib import dump
import os

#load the dataset
df=pd.read_csv('C:/Users/a6shl/downloads/DoS_dataset.csv',names=['time', 'ID', 'DLC','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Label'])
df.dropna(inplace=True)
df.dropna(subset=["Label"], inplace=True)

#data labeling; replace r --> 1, t --> 0 
df['Label'].replace({'R':1, 'T':0}, inplace = True)

#data conversion from hexadecimal --> integer
df['Data0']=df['Data0'].apply(lambda x:int(x, 16))
df['Data1']=df['Data1'].apply(lambda x:int(x, 16))
df['Data2']=df['Data2'].apply(lambda x:int(x, 16))
df['Data3']=df['Data3'].apply(lambda x:int(x, 16))
df['Data4']=df['Data4'].apply(lambda x:int(x, 16))
df['Data5']=df['Data5'].apply(lambda x:int(x, 16))
df['Data6']=df['Data6'].apply(lambda x:int(x, 16))
df['Data7']=df['Data7'].apply(lambda x:int(x, 16))
df['ID']=df['ID'].apply(lambda x:int(x, 16))

# Sample the data
sample_df = df.sample(frac=0.0015, random_state=55)  # Adjust fraction as needed

# Extract features and labels
y = sample_df[['Label']].copy()  # Predictor variable
X = sample_df[['ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']].copy()  # Features

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model 
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train.values.ravel())

# Model predictions on test set
model_predictions = model.predict(X_test_scaled) 

# Accuracy test
accuracy_test = accuracy_score(y_test, model_predictions)

# Cross-validation on the training set
cv_scores = cross_val_score(model, X_train_scaled, y_train.values.ravel(), cv=6)

# Classification report
report = classification_report(y_test, model_predictions)

print("Accuracy:", accuracy_test)
print("Report:\n", report)
print("Cross validation: \n", cv_scores)


# Define the file path for saving the model
model_file_path = 'C:/Users/a6shl/OneDrive/Documents/GitHub/carzz/knn_model.joblib'

# Save the trained model
dump(model, model_file_path)  # Save the model to the specified file path
