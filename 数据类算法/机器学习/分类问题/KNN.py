from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv("iris.csv")
X = df[['X1', 'X2', 'X3', 'X4']]
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# train regressor
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
# evaluate
print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test, y_predict))
