import matplotlib.pyplot as plt
import numpy as np

dataset = {f'experience {i}' : np.random.randn(100,3) for i in range(6)}

def graphique(data):
    n = len(data)
    plt.figure(figsize=(12,30))

    for k,i in zip(data.keys(), range(1,n+1)):
        plt.subplot(n,1,i)
        plt.plot(data[k])
        plt.title(k)
    plt.show()

graphique(dataset)
