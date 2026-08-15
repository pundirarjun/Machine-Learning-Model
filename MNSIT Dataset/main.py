import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'D:\Machine Learning Model\MNSIT Dataset\train.csv')
X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train , X_test, y_train , y_test = train_test_split(X,y, test_size= 0.2 , random_state= 0)


results = []

for i in range(80,100):
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=i)),
        ('KNN',KNeighborsClassifier(n_neighbors=5 , n_jobs= -1))
    ])

    pipeline.fit(X_train, y_train )

    y_pred = pipeline.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    pca = pipeline.named_steps['PCA']

    

    results.append([i,score])

    # print(f'PCA Componenets : {i:2d} | Accuracy: {score:.4f}')

    results_df = pd.DataFrame(
        results,
        columns = ['PCA_Components', 'Accuracy']
    )
    pass

# print(results_df)


# plt.plot(
#     results_df['PCA_Components'],
#     results_df['Accuracy'],
#     marker='o'
# )

# plt.xlabel('Number of PCA Components')
# plt.ylabel('Accuracy')
# plt.title('PCA Components vs KNN Accuracy')

# plt.grid()
# plt.show()


plt.plot(np.cumsum(pca.explained_variance_ratio_))
