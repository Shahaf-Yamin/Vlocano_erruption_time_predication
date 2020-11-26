import pandas as pd
import seaborn as sns
import os
import numpy as np

class data_analyzer(object):
    def __init__(self,data,config, name):
        self.frags = data['frags']
        self.segments = data['segments']
        self.config = config
        self.__name__ = name
        self.path = 'data/features'
        self.feature_set = None

    def analyse_missing_observation(self):
        sensors = set()
        observations = set()
        nan_columns = list()
        missed_groups = list()
        for_df = list()
        self.config.dataset_size = len(self.frags) if self.config.dataset_size == 'max' else self.config.dataset_size
        for cnt,item in enumerate(self.frags):
            if cnt > self.config.dataset_size:
                break
            # extract the segment name
            name = int(item.split('.')[-2].split('/')[-1])
            at_least_one_missed = 0
            frag = pd.read_csv(item)
            missed_group = list()
            missed_percents = list()
            # Run the segment columns
            for col in frag.columns:
                # calculate how many values are missing in percentages from this column
                missed_percents.append(frag[col].isnull().sum() / len(frag))
                #check if we are missing at list a sensor sample
                if pd.isnull(frag[col]).all() == True:
                    at_least_one_missed = 1
                    nan_columns.append(col)
                    missed_group.append(col)
            if len(missed_group) > 0:
                missed_groups.append(missed_group)
            sensors.add(len(frag.columns))
            observations.add(len(frag))
            for_df.append([name, at_least_one_missed] + missed_percents)

        self.for_df = pd.DataFrame(for_df,columns=['segment_id', 'has_missed_sensors', 'missed_percent_sensor1',
                'missed_percent_sensor2', 'missed_percent_sensor3', 'missed_percent_sensor4',
                'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7',
                'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10'])

        self.merged = pd.merge(self.segments, self.for_df)

        if self.config.with_missing_sensor_dist_analysis:
            self.analyse_missing_sensor_distribution(nan_columns,missed_groups)

    def analyse_missing_sensor_distribution(self,nan_columns,missed_groups):
        absent_sensors = dict()
        for item in nan_columns:
            if item in absent_sensors:
                absent_sensors[item] += 1
            else:
                absent_sensors[item] = 0

        absent_df = pd.DataFrame(absent_sensors.items(), columns=['Sensor', 'Missed sensors'])

        # fig = px.bar(
        #     absent_df,
        #     x="Sensor",
        #     y='Missed sensors',
        #     width=800,
        #     height=500,
        #     title='Number of missed sensors in training dataset'
        # )
        # fig.show()

        absent_groups = dict()

        for item in missed_groups:
            if str(item) in absent_groups:
                absent_groups[str(item)] += 1
            else:
                absent_groups[str(item)] = 0

    def analyse_segements_sensors_correlation(self):
        if self.config.pairplot_sensors_correaltion is True:
            indices = np.random.randint(len(self.train_frags), size=self.config.pairplot_number_of_sensors)
            for index in indices:
                item = self.train_frags[index]
                name = int(item.split('.')[-2].split('/')[-1])
                frag = pd.read_csv(item)
                sns_plot = sns.pairplot(frag)
                if not os.path.exists('outputs/pair_plots'):
                    os.makedirs('outputs/pair_plots')
                sns_plot.savefig(f"outputs/pair_plots/segment_{name}.png")

    def build_features_signal(self, signal, ts, sensor_id):
        X = pd.DataFrame()
        f = np.fft.fft(signal)
        f_real = np.real(f)
        f_imag = np.imag(f)
        f_abs = np.abs(f)
        X.loc[ts, f'{sensor_id}_sum'] = signal.sum()
        X.loc[ts, f'{sensor_id}_mean'] = signal.mean()
        X.loc[ts, f'{sensor_id}_std'] = signal.std()
        X.loc[ts, f'{sensor_id}_var'] = signal.var()
        X.loc[ts, f'{sensor_id}_max'] = signal.max()
        X.loc[ts, f'{sensor_id}_min'] = signal.min()
        X.loc[ts, f'{sensor_id}_skew'] = signal.skew()
        X.loc[ts, f'{sensor_id}_mad'] = signal.mad()
        X.loc[ts, f'{sensor_id}_kurtosis'] = signal.kurtosis()
        X.loc[ts, f'{sensor_id}_quantile99'] = np.quantile(signal, 0.99)
        X.loc[ts, f'{sensor_id}_quantile95'] = np.quantile(signal, 0.95)
        X.loc[ts, f'{sensor_id}_quantile85'] = np.quantile(signal, 0.85)
        X.loc[ts, f'{sensor_id}_quantile75'] = np.quantile(signal, 0.75)
        X.loc[ts, f'{sensor_id}_quantile55'] = np.quantile(signal, 0.55)
        X.loc[ts, f'{sensor_id}_quantile45'] = np.quantile(signal, 0.45)
        X.loc[ts, f'{sensor_id}_quantile25'] = np.quantile(signal, 0.25)
        X.loc[ts, f'{sensor_id}_quantile15'] = np.quantile(signal, 0.15)
        X.loc[ts, f'{sensor_id}_quantile05'] = np.quantile(signal, 0.05)
        X.loc[ts, f'{sensor_id}_quantile01'] = np.quantile(signal, 0.01)
        X.loc[ts, f'{sensor_id}_fft_real_mean'] = f_real.mean()
        X.loc[ts, f'{sensor_id}_fft_real_std'] = f_real.std()
        X.loc[ts, f'{sensor_id}_fft_real_max'] = f_real.max()
        X.loc[ts, f'{sensor_id}_fft_real_min'] = f_real.min()
        X.loc[ts, f'{sensor_id}_fft_imag_mean'] = f_imag.mean()
        X.loc[ts, f'{sensor_id}_fft_imag_std'] = f_imag.std()
        X.loc[ts, f'{sensor_id}_fft_imag_max'] = f_imag.max()
        X.loc[ts, f'{sensor_id}_fft_imag_min'] = f_imag.min()
        X.loc[ts, f'{sensor_id}_fft_abs_mean'] = f_abs.mean()
        X.loc[ts, f'{sensor_id}_fft_abs_std'] = f_abs.std()
        X.loc[ts, f'{sensor_id}_fft_abs_max'] = f_abs.max()
        X.loc[ts, f'{sensor_id}_fft_abs_min'] = f_abs.min()
        return X

    def extract_data_features(self):
        self.feature_set = list()
        j = 0
        for seg in self.merged.segment_id:
            # read singls from csv
            signals = pd.read_csv(f'data/dataset/{self.__name__}/{seg}.csv')
            train_row = []
            if j % 500 == 0:
                print(j)
            for sensor_number in range(0, 10):
                #iterate over all sensor's signal
                sensor_id = f'sensor_{sensor_number + 1}'
                # TODO: find how to fill the missing values!!!
                train_row.append(self.build_features_signal(signals[sensor_id].fillna(0), seg, sensor_id))
            train_row = pd.concat(train_row, axis=1)
            self.feature_set.append(train_row)
            j += 1
        self.feature_set = pd.concat(self.feature_set)
        self.feature_set = self.feature_set.reset_index()
        self.feature_set = self.feature_set.rename(columns={'index': 'segment_id'})

        self.feature_set = pd.merge(self.feature_set, self.merged, on='segment_id')


        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.feature_set.to_csv(f'{self.path}/{self.config.feature_version}_{self.__name__}_with_redudant_feat.csv', index=False)

    def divide_input_output(self):
        if self.__name__ == 'train':
            X = self.feature_set.drop(['segment_id', 'time_to_eruption'], axis=1)
            y = self.feature_set['time_to_eruption']
            return X, y
        elif self.__name__ == 'test':
            X = self.feature_set.drop(['segment_id', 'time_to_eruption'], axis=1)
            return X
        else:
            raise Exception('Invalid class name please use train or test or define a new name!')

    def remove_redudant_features(self):
        if self.feature_set is None:
            raise Exception('Invalid command, please extract or load features in order to remove redudant ones!')
        drop_cols = []
        # Remove uncorrelated columns
        for col in self.feature_set.columns:
            if col == 'segment_id':
                continue
            if abs(self.feature_set[col].corr(self.feature_set['time_to_eruption'])) < self.config.uncorrelated_cols_threshold:
                drop_cols.append(col)

        not_to_drop_cols = []

        for col1 in self.feature_set.columns:
            for col2 in self.feature_set.columns:
                if col1 == col2:
                    continue
                if col1 == 'segment_id' or col2 == 'segment_id':
                    continue
                if col1 == 'time_to_eruption' or col2 == 'time_to_eruption':
                    continue
                if abs(self.feature_set[col1].corr(self.feature_set[col2])) > self.config.correlated_cols_threshold:
                    if col2 not in drop_cols and col1 not in not_to_drop_cols:
                        drop_cols.append(col2)
                        not_to_drop_cols.append(col1)
        self.feature_set.to_csv(f'{self.path}/{self.config.feature_version}_{self.__name__}_without_redudant_feat.csv', index=False)
        return self.divide_input_output()

    def load_data_features_before_removing_features(self):
        if not os.path.exists(self.path):
            raise Exception('Missing features! please extract them first')
        self.feature_set = pd.read_csv(f'{self.path}/{self.config.feature_version}_{self.__name__}_with_redudant_feat.csv')
        return self.divide_input_output()

    def load_data_features_after_removing_features(self):
        if not os.path.exists(self.path):
            raise Exception('Missing features! please extract them first')
        self.feature_set = pd.read_csv(f'{self.path}/{self.config.feature_version}_{self.__name__}_without_redudant_feat.csv')
        return self.divide_input_output()