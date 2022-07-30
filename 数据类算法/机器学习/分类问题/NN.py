from sklearn.datasets import load_iris
X,y=load_iris(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# First 声明神经网络回归函数
from sklearn.neural_network import MLPClassifier
NNmodel = MLPClassifier([100,90,80,70,60,50,40,30,20,10,5,3],learning_rate_init= 0.001,activation='tanh',\
     solver='adam', alpha=0.01,max_iter=3000000)  # 神经网络
#Second 训练数据
print('start train!')
NNmodel.fit(X_train,y_train)
print('end train!')
#Third 检验训练集的准确性
y_predict=NNmodel.predict(X_test)
#evaluate
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))
