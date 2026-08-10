import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

df = pd.read_csv(r'D:\Machine Learning Model\Iris datasets\IRIS.csv')

X = df.iloc[:,:4]
y = df.iloc[:,4]

X_train , X_test, y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(accuracy_score(y_test, y_pred))
