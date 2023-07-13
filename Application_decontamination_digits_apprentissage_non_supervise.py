from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis


digits = load_digits()
images = digits.images
X = digits.data
y = digits.target

print(X.shape)
# model = IsolationForest(random_state=0,contamination=0.2)
# model.fit(X)
# Ici c'est la methode d'isolation
# outliers = model.predict(X) == -1
# donne_recup = images[outliers][0]
# plt.title(Y[outliers][0])
# plt.imshow(donne_recup)
# plt.show()


# Reduction de donnee

model = PCA(n_components=64)
X_reduced = model.fit_transform(X)

np.cumsum(model.explained_variance_ratio_)
# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=Y)
# plt.colorbar()
plt.show()