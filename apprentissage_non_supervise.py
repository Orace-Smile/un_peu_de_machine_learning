#K-Means Clustering  Algorithme
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_blobs
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# X,y = make_blobs(n_samples=100,centers=3,cluster_std=1.0)
#
# model = KMeans(n_clusters=3)
# model.fit(X)
# model.predict(X)
# plt.scatter(X[:,0],X[:,1],c=model.predict(X))
# plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],c='r')
# print(model.cluster_centers_)
# plt.show()

# Detection d'anomalie
# X,y = make_blobs(n_samples=50,centers=1,cluster_std=0.2)
# X[-1,:] = np.array([2.25,5])
# model = IsolationForest(contamination=0.01)
# model.fit(X)
# plt.scatter(X[:,0],X[:,1],c=model.predict(X))
# plt.show()

""" 
Le bagging boosting et le stacking
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X,y = make_moons(n_samples=500,noise=0.3,random_state=0)
plt.scatter(X[:,0],X[:,1],c=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

model_1 = SGDClassifier(random_state=0)
model_2 = DecisionTreeClassifier(random_state=0)
model_3 = KNeighborsClassifier(n_neighbors=2)

model_4 = VotingClassifier([('SGD',model_1),
                            ('Tree',model_2),
                            ('KNN',model_3)],voting='hard')

# for model in (model_1, model_2, model_3,model_4):
#     model.fit(X_train, y_train)
#     print(model.__class__.__name__,model.score(X_test,y_test))

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

model = BaggingClassifier(estimator= KNeighborsClassifier(),n_estimators=100)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))