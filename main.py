
import numpy as np
import pandas as pd
import pickle
df=pd.read_csv("diabetes.csv")
#print(df.head(30))
#print("null values\n",df.isnull().sum())
#print("shape\n",df.shape)
#print("datatypes\n",df.dtypes)
#df.info()
#print(df.describe())
from sklearn import preprocessing
##print(df)
# remove 'Other' gender
##df = df[df['gender'] != 'Other']
# now check counts
##print(df["gender"].value_counts())
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
Classifier=RandomForestClassifier(n_estimators=20)
Classifier.fit(X_train,y_train)
##pickle file
filename="diabetes-prediction(rfc).pkl"
pickle.dump(Classifier,open(filename,"wb"))




