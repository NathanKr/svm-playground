from utils import load_dataset
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


X,y,x1,x2,m,ar_index_pass,ar_index_fail = load_dataset("ex6data2.mat")

def show():
    plt.title(f'data set : + : pass , o : fail')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1[ar_index_pass],x2[ar_index_pass],'+')
    plt.plot(x1[ar_index_fail],x2[ar_index_fail],'o')
    plt.grid()
    plt.show()

def train(C):
    """[summary]

    Args:
        C ([type]): C = 1 / lambda the regularization factor
        lambda is 0   => C is big => no regularization and thus may high bias and overfit
        lambda is big => C is small => regularization and thus may low bias and nooverfit
    """
    model = svm.SVC( C=C) # default kernel is guasian
    y_fixed_for_fit = y.ravel() # required by model.fit
    return model.fit(X,y_fixed_for_fit) 

# show()
C=1
model_fitted = train(C) 

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model_fitted.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,  alpha=0.8)

# Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.title(titles[i])
plt.show()

# # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = model_fitted.predict(X)

# # Put the result into a color plot
# Z = Z.reshape(X.shape)
# # plt.contourf(X, y, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.contourf(X, y, Z)