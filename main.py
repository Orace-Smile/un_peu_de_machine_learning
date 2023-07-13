import random

import numpy as np
#
# A = np.array([1,2,3])
# B = np.zeros((2,3))
# C = np.ones((4,4))
# D = np.random.randn(3,5)
# print(B)
# print('-----------------------------------')
# print(C)
# print('---------------------------------')
# print(D)

# def initialisation(m,n):
#     n = np.concatenate((m,n+1),axis=2)
#     matrice = np.random.randn((m,n))
#     return matrice
#
# initialisation(3,4)

# A = np.array([[1,2,5],[5,9,4],[4,5,7]])
#
# print(A)
# print("------------------")
#
# B = A[0:3,1:3]
# print(B)
# print("----------------")
# C = np.zeros((4,4))
# print(C)

from scipy import misc
import matplotlib.pyplot as plt
face = misc.face()
plt.imshow(face,cmap=plt.cm.gray)
plt.show()
print(face.shape)