import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle

df = pd.read_csv("data_moods.csv")

print(df.columns)
print(df.info())

# 'popularity', 'release_date', 'id', 'duration_ms',
# 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
# 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
# 'valence', 'tempo'

X, y = df[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness',
       'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key']], df["mood"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)

predict = rfc.predict(X_test)

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

pickle.dump(rfc, open("moods.pkl", "wb"))