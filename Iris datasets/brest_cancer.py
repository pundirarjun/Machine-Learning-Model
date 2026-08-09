import pandas as pd
from sklearn import pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.iloc[:, :30]
y = df.iloc[:, -1]

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 7, weights='distance',p=2, metric = 'minkowski',n_jobs= -1)

knn.fit(X_train ,y_train )
y_pred = knn.predict(X_test)

print(f'KNN accuracy score: ',accuracy_score(y_test,y_pred))

# best_k = 1
# best_score = 0

# for k in range(1,21):
#
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('knn', KNeighborsClassifier(n_neighbors=k,weights='distance',p=2, metric = 'minkowski',n_jobs= -1)),
#         ('Decision Tree', DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0))
#     ])
#
#     scores = cross_val_score(pipeline , X, y , cv = 5)
#
#     mean_score = scores.mean()
#
#     print(f"K = {k:2d} | Scores = {scores} | Mean = {mean_score:.4f}")
#
#     if mean_score > best_score:
#         best_score = mean_score
#         best_k = k
#
# print("\nBest K:", best_k)
# print("Best Accuracy:", best_score)


dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)
dt.fit(X_train , y_train )
y_pred1 = dt.predict(X_test)

print(f'Decision Tree accuracy score: ',accuracy_score(y_test, y_pred1))

print(f'Confusion metric for decision tree: \n',confusion_matrix(y_test, y_pred1))
print(f'Confusion metric for KNN: \n',confusion_matrix(y_test, y_pred))