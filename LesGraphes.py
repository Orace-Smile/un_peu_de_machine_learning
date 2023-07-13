import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
#
x = np.linspace(0,4,10)
# y = x**3
# print(x)
# plt.figure()
# plt.plot(x,y,ls=":")
# plt.plot(x,x**5,ls="--")
# plt.title("Graphe 1")
# # plt.plot(x,y)
# plt.show()
iris = load_iris()
x = iris.data
y = iris.target
names = list(iris.target_names)
# Pour faire un graphique en 2d
# plt.scatter(x[:,0],x[:,1] ,c=y)
# Pour faire un graphique en 3d
# ax = plt.axes(projection='3d')
# ax.scatter(x[:,0],x[:,1],x[:,2],c=y)
# f = lambda x, y: np.sin(x) + np.cos(x+y)
#
# X = np.linspace(0,5,100)
# Y = np.linspace(0,5,100)
# X, Y = np.meshgrid(X,Y)
#
# Z = f(X,Y)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X,Y,Z,cmap="Greens")
# plt.show()
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}

def graphique(data):
    n = len(data)
    plt.figure(figsize=(12,8))

    for k, i in zip(data.keys(),range(1,n+1)):
        plt.subplot(n,1,i)
        plt.plot(data[k])
        plt.title(k)
    plt.show()

graphique(dataset)



