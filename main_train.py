from trainers.trainer import train
from data.data_loader import load_data,preprocess_data
from models.models import build_model
from utils.utils import preprocess_meta_data

import pandas as pd

#tf.keras.backend.set_floatx('float32')

def main():
    # capture the config path from the run arguments
    # then process configuration file
    config = preprocess_meta_data()

    if not config.quiet:
        config.print()

    # load the data
    data,test_segment_id = load_data(config)

    # preprocess data before training

    data = preprocess_data(data,config)

    # create a model
    model = build_model(config)

    # create trainer
    p_test = train(model, data, config)

    submission_save = pd.DataFrame()
    submission_save['segment_id'] = test_segment_id
    submission_save['time_to_eruption'] = p_test
    submission_save.to_csv(f'{config.exp_name}.csv', header=True, index=False)


if __name__ == '__main__':
    main()









