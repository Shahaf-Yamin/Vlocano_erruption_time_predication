from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Softmax
#from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge


import numpy as np

def build_model(config):
    # if config.model_name == "LGBM":
    #     model = {'train': LGBM(config),
    #              'eval':  LGBM(config)}
    if config.model_name == 'ridge':
        model = {'train': ridge_model(config),
                 'eval': ridge_model(config)}
    elif config.model_name == 'kernel_ridge':
        model = {'train': kernel_ridge_model(config),
                 'eval': kernel_ridge_model(config)}
    elif config.model_name == 'kernel_knn':
        model = {'train': kernel_knn_model(config),
                'eval': kernel_knn_model(config)}
    else:
        raise ValueError("'{}' is an invalid model name")

    return model
def kernel_knn_model(config):
    kernel_width = config.kernel_ridge_gamma
    def gaussian_kernel(distances):
        non_loc_kernel_width = kernel_width
        weights = np.exp(-(distances**2)/non_loc_kernel_width)
        return weights

    model = KNeighborsRegressor(n_neighbors=10, weights=gaussian_kernel)
    return model

def kernel_ridge_model(config):
    model = KernelRidge(kernel='rbf', alpha=config.kernel_ridge_alpha, gamma=config.kernel_ridge_gamma)
    return model

def ridge_model(config):
    model = Ridge(alpha=config.alpha, normalize=True)
    return model
