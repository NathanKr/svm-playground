from utils import load_dataset
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


X,y,x1,x2,m,ar_index_pass,ar_index_fail = load_dataset("ex6data2.mat")

def plot_dataset(title):
    plt.title(title)
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

def plot_contour(model_fitted):
    margin = 0.2
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    H = model_fitted.predict(np.c_[xx.ravel(), yy.ravel()])
    H = H.reshape(xx.shape)
    plt.contour(xx, yy, H) # H is either 1 or 0

def run(C):
    model_fitted = train(C) 
    plot_contour(model_fitted)
    plot_dataset(f'data set : + : pass , o : fail vs SVM nonlinear decision boundary C = {C} (guasian kernel )')

plot_dataset(f'data set : + : pass , o : fail')
run(C=1)
run(C=100) # overfit