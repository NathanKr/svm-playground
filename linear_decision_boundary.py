from os.path import join 
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,"ex6data1.mat")
mat_dict = sio.loadmat(file_name)
# print("mat_dict.keys() : ",mat_dict.keys())

X = mat_dict["X"]
x1 = X[:,0]
x2 = X[:,1]
y = mat_dict["y"]
m = y.size

ar_index_pass = np.where(y == 1)[0]
ar_index_fail = np.where(y == 0)[0]


def show(model_fitted):
    print(f"coef_ : {model_fitted.coef_}")
    print(f"intercept_ : {model_fitted.intercept_}")
    h = model_fitted.predict(X) # same as sigmoid(intercept_[0]+coef_[0]*x1+coef_[1]*x2)
    count_wrong=0
    for it in h == y.ravel():
        if it == False:
            count_wrong += 1

    print(f"score : {100*(m-count_wrong)/m} [%]")

    plt.title(f'data set : + : pass , o : fail \nstraight line is the linear SVM decision boundry with C = {model_fitted.C}')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1[ar_index_pass],x2[ar_index_pass],'+')
    plt.plot(x1[ar_index_fail],x2[ar_index_fail],'o')

    # for decision boundary for sigmoid we take 0=intercept_[0]+coef_[0]*x1+coef_[1]*x2
    # thus x2 = -(intercept_[0]+coef_[0]*x1) / coef_[1]
    decision_boundry = -(model_fitted.intercept_[0]+model_fitted.coef_[0,0]*x1) / model_fitted.coef_[0,1]
    plt.plot(x1,decision_boundry)

    plt.grid()
    plt.show()

def train(C):
    """[summary]

    Args:
        C ([type]): C = 1 / lambda the regularization factor
        lambda is 0   => C is big => no regularization and thus may high bias and overfit
        lambda is big => C is small => regularization and thus may low bias and nooverfit
    """
    model = svm.SVC(kernel='linear', C=C)# 
    y_fixed_for_fit = y.ravel() # required by model.fit
    return model.fit(X,y_fixed_for_fit)  

C = 100 # this overfit
model_fitted = train(C) 
show(model_fitted) 

C = 1 # overfit fixed with regularization`
model_fitted = train(C) 
show(model_fitted)  

