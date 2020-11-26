from trainers.trainer import build_trainer
from data.data_loader import load_data
from models.models import build_model
from utils.utils import preprocess_meta_data
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.keras.backend.set_floatx('float32')

def main():
    # capture the config path from the run arguments
    # then process configuration file
    config = preprocess_meta_data()

    # load the data
    data = load_data(config)
    X, y = data['train']
    X_train, X_val, y, y_val = train_test_split(X, y, random_state=666, test_size=0.2, shuffle=True)

    if not config.quiet:
        config.print()

    # create a model
    model = build_model(config)

    # create trainer
    trainer = build_trainer(model, data, config)

    # train the model
    trainer.train()


if __name__ == '__main__':
    main()









