import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


#load the dataset
df=pd.read_csv('C:/Users/a6shl/downloads/DoS_dataset.csv',names=['time', 'ID', 'DLC','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Label'])
df.dropna(inplace=True)
df.dropna(subset=["Label"], inplace=True)
# print(df.tail())

#data labeling; replace r --> 1, t --> 0 
df['Label'].replace({'R':1, 'T':0}, inplace = True)
# inplace = instead of making new column, just replace

#data conversion from hexadecimal --> integer
df['Data0']=df['Data0'].apply(lambda x:int(x, 16)) # accessing column, use apply function to complete conversion
df['Data1']=df['Data1'].apply(lambda x:int(x, 16))
df['Data2']=df['Data2'].apply(lambda x:int(x, 16))
df['Data3']=df['Data3'].apply(lambda x:int(x, 16))
df['Data4']=df['Data4'].apply(lambda x:int(x, 16))
df['Data5']=df['Data5'].apply(lambda x:int(x, 16))
df['Data6']=df['Data6'].apply(lambda x:int(x, 16))
df['Data7']=df['Data7'].apply(lambda x:int(x, 16))
df['ID']=df['ID'].apply(lambda x:int(x, 16))

# print(df[df['Label']==0]) --> to see t values 

sample_df = df.sample(frac=0.02, random_state=42)  #10.8k values instead 

 #split the sampled data into test/train
y = sample_df[['Label']].copy()  #predictor variable
X = sample_df[['ID', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7']].copy()  # Feature

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


scaler=MinMaxScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#create model 
model = KNeighborsClassifier()
model.fit(X_trained_scaled, y_train.values.ravel())

#model
model_predictions = model.predict(X_test_scaled)

#accuracy test
accuracy_test = accuracy_score(y_test, model_predictions)
print("Accuracy:", accuracy_test)

report = classification_report(y_test, model_predictions)
print("Report:\n", report)