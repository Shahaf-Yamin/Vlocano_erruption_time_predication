from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Softmax
from lightgbm import LGBMRegressor

def build_model(config):
    if config.model_name == "LGBM":
        model = {'train': LGBM(config),
                 'eval':  LGBM(config)}

    else:
        raise ValueError("'{}' is an invalid model name")

    return model

def LGBM(config):
    return LGBMRegressor(random_state=666, max_depth=config.LGBM_max_depth,
                        n_estimators=config.LGBM_number_of_estimators, learning_rate=config.learning_rate)

def CNNModel(config):

    model = Sequential()
    model.add(Conv2D(config.hidden_size[0], (3, 3), activation='relu', input_shape=config.model_input_dim))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(config.hidden_size[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(config.model_output_dim, activation=None))
    model.add(Softmax(axis=-1))

    return model
