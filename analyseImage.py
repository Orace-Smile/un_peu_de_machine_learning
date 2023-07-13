import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
print('Train set', X_train.shape)
print('Train test',X_test.shape)
print(X.shape)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train, alpha=0.8)
plt.title('Train set')
plt.subplot(122)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,alpha=0.8)
plt.title('Test set')

# plt.scatter(X[:,0],X[:,1],c=y,alpha=0.8)
# plt.show()
model = KNeighborsClassifier(n_neighbors=2)
# model.fit(X_train,y_train)
k = np.arange(1,50)
train_score, val_score = validation_curve(model,X_train,y_train)
print('Train set',model.score(X_train,y_train))
print('Test set',model.score(X_test,y_test))
# plt.show()


