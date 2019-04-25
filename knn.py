from sklearn.neighbors import KNeighborsClassifier
x = [[0],[1],[2],[3],[4],[5]]
y = [0,1,1,0,1,0]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)
print(knn.predict([[0.9]]))
print(knn.predict_proba([[4]]))