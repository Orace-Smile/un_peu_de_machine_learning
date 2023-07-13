import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(0)
m = 100
X = np.linspace(0,10,m).reshape(m,1)
y = X + np.random.randn(m,1)
model = LinearRegression()
model.fit(X,y)
print(model.score(X,y))
prediction = model.predict(X)
plt.scatter(X,y)
plt.plot(X,prediction,c='r')
plt.show()