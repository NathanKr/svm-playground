from os.path import join 
import os
import scipy.io as sio
import numpy as np

def load_dataset(file):
    current_dir = os.path.abspath(".")
    data_dir = join(current_dir, 'data')
    file_name = join(data_dir,file)
    mat_dict = sio.loadmat(file_name)
    # print("mat_dict.keys() : ",mat_dict.keys())

    X = mat_dict["X"]
    x1 = X[:,0]
    x2 = X[:,1]
    y = mat_dict["y"]
    m = y.size

    ar_index_pass = np.where(y == 1)[0]
    ar_index_fail = np.where(y == 0)[0]
    
    return(X,y,x1,x2,m,ar_index_pass,ar_index_fail)



