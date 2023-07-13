from utilities import *
from premier_programme import *
X_train, y_train, X_test, y_test = load_data()

print(X_train.shape)
print(y_train.shape)
print(np.unique(y_test,return_counts= True))

W , b = artificial_neuron(X_train,y_train)

