import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KKNeighboursClassifiers import KNN

df = pd.read_csv('Social_Network_Ads.csv')

df = df.iloc[:,1:]
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
scaler = StandardScaler()
X = df.iloc[:,0:3].values
X = scaler.fit_transform(X)
y = df.iloc[:,-1].values

X_train , X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 0 )

knn =  KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print( accuracy_score(y_test, y_pred) )

apnaknn = KNN(k=5)
apnaknn.fit(X_train,y_train)
y_pred1 = apnaknn.predict(X_test)

print(accuracy_score(y_test, y_pred1))



