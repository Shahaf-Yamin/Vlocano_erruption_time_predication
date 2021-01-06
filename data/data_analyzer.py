import pandas as pd
import seaborn as sns
import os
import numpy as np
from scipy import signal as sig_lib
import sklearn.decomposition as decomposition
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

class data_analyzer(object):
    def __init__(self,data,config, name, psd_scaler=None,pca_transformer=None):
        self.frags = data['frags']
        self.segments = data['segments']
        self.config = config
        self.__name__ = name
        self.path = 'data/features'
        self.feature_set = None

        self.psd_pca_scaler = psd_scaler
        self.psd_pca_transformer = pca_transformer

    def analyse_missing_observation(self):
        sensors = set()
        observations = set()
        nan_columns = list()
        missed_groups = list()
        for_df = list()
        datset_size = len(self.frags) if self.config.dataset_size == 'max' else self.config.dataset_size
        cnt = 0
        for item in self.frags:
            if cnt > datset_size:
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
                #check if we are missing at least one sample
                if pd.isnull(frag[col]).all() == True:
                    at_least_one_missed = 1
                    nan_columns.append(col)
                    missed_group.append(col)
            if len(missed_group) > 0:
                missed_groups.append(missed_group)
            sensors.add(len(frag.columns))
            observations.add(len(frag))
            for_df.append([name, at_least_one_missed] + missed_percents)
            cnt += 1

        self.for_df = pd.DataFrame(for_df, columns=['segment_id', 'has_missed_sensors', 'missed_percent_sensor1',
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

        fs = 100  # sampling frequency
        freq, psd = sig_lib.welch(signal, fs=fs)




        if signal.isna().sum() > 1000:  ##########
            X.loc[ts, f'{sensor_id}_A_pow']  = np.nan
            X.loc[ts, f'{sensor_id}_A_num']  = np.nan
            X.loc[ts, f'{sensor_id}_BH_pow'] = np.nan
            X.loc[ts, f'{sensor_id}_BH_num'] = np.nan
            X.loc[ts, f'{sensor_id}_BL_pow'] = np.nan
            X.loc[ts, f'{sensor_id}_BL_num'] = np.nan
            X.loc[ts, f'{sensor_id}_C_pow']  = np.nan
            X.loc[ts, f'{sensor_id}_C_num']  = np.nan
            X.loc[ts, f'{sensor_id}_D_pow']  = np.nan
            X.loc[ts, f'{sensor_id}_D_num']  = np.nan
            X.loc[ts, f'{sensor_id}_fft_real_mean'] = np.nan
            X.loc[ts, f'{sensor_id}_fft_real_std'] = np.nan
            X.loc[ts, f'{sensor_id}_fft_real_max'] = np.nan
            X.loc[ts, f'{sensor_id}_fft_real_min'] = np.nan
        else:
            # STFT(Short Time Fourier Transform) Specifications
            n = 256  # FFT segment size
            max_f = 20  # ～20Hz

            delta_f = fs / n  # 0.39Hz
            # delta_t = n / fs / 2  # 1.28s
            f = np.fft.fft(signal)
            f_real = np.real(f)
            f, t, Z = sig_lib.stft(signal.fillna(0), fs=fs, window='hann', nperseg=n)
            # f = f[:round(max_f / delta_f) + 1]
            Z = np.abs(Z[:round(max_f / delta_f) + 1]).T  # ～max_f, row:time,col:freq

            th = Z.mean() * 1  ##########
            Z_pow = Z.copy()
            Z_pow[Z < th] = 0
            Z_num = Z_pow.copy()
            Z_num[Z >= th] = 1

            Z_pow_sum = Z_pow.sum(axis=0)
            Z_num_sum = Z_num.sum(axis=0)

            X.loc[ts, f'{sensor_id}_A_pow']  = Z_pow_sum[round(10 / delta_f):].sum()
            X.loc[ts, f'{sensor_id}_A_num']  = Z_num_sum[round(10 / delta_f):].sum()
            X.loc[ts, f'{sensor_id}_BH_pow'] = Z_pow_sum[round(5 / delta_f):round(8 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_BH_num'] = Z_num_sum[round(5 / delta_f):round(8 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_BL_pow'] = Z_pow_sum[round(1.5 / delta_f):round(2.5 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_BL_num'] = Z_num_sum[round(1.5 / delta_f):round(2.5 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_C_pow']  = Z_pow_sum[round(0.6 / delta_f):round(1.2 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_C_num']  = Z_num_sum[round(0.6 / delta_f):round(1.2 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_D_pow']  = Z_pow_sum[round(2 / delta_f):round(4 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_D_num']  = Z_num_sum[round(2 / delta_f):round(4 / delta_f)].sum()
            X.loc[ts, f'{sensor_id}_fft_real_mean'] = f_real.mean()
            X.loc[ts, f'{sensor_id}_fft_real_std'] = f_real.std()
            X.loc[ts, f'{sensor_id}_fft_real_max'] = f_real.max()
            X.loc[ts, f'{sensor_id}_fft_real_min'] = f_real.min()
        return X, psd

    def calculate_statistics(self):
        self.feature_set = list()
        j = 0
        print('Starting statistics calculation')
        signal_record_dict = {}
        '''
        Initialize the sensor statistics
        '''
        for sensor_number in range(0, 10):
            # iterate over all sensor's signals
            sensor_id = f'sensor_{sensor_number + 1}'
            signal_record_dict[sensor_id] = np.array([])

        for seg in self.merged.segment_id:
            # read signals from csv
            signals = pd.read_csv(f'data/dataset/{self.__name__}/{seg}.csv')
            train_row = []
            if j % 500 == 0:
                print(j)
            for sensor_number in range(0, 10):
                # iterate over all sensor's signals
                sensor_id = f'sensor_{sensor_number + 1}'
                if signals[sensor_id].isnull().values.any():
                    # check if there are NaN values
                    if signals[sensor_id].isnull().sum() != len(signals[sensor_id]):
                        np.concatenate([signal_record_dict[sensor_id], signals[sensor_id].dropna().values], axis=0)
                else:
                    signal_record_dict[sensor_id] = np.concatenate([signal_record_dict[sensor_id], signals[sensor_id].values], axis=0)
            j = j + 1

        self.statistic = {'mean': {}, 'std': {}}
        for sensor_number in range(0, 10):
            # iterate over all sensor's signals
            sensor_id = f'sensor_{sensor_number + 1}'
            self.statistic['mean'][sensor_id] = np.mean(signal_record_dict[sensor_id])
            self.statistic['std'][sensor_id] = np.std(signal_record_dict[sensor_id])

    def fill_missing_values(self,signal,sensor_id):
        if signal.isnull().values.any():
            if signal.isnull().sum() != len(signal):
                # this case we will use linear interpolation:
                signal = signal.mode(dropna=True)
            else:
                mean = self.statistic['mean'][sensor_id]
                std = self.statistic['std'][sensor_id]
                random_sample = std * np.random.normal() + mean
        else:
            # this case there are no missing values
            return signal

    def extract_data_features(self):
        self.feature_set = list()
        psd_samples = list()
        j = 0
        for seg in self.merged.segment_id:
            # read signals from csv
            signals = pd.read_csv(f'data/dataset/{self.__name__}/{seg}.csv')
            train_row = []
            if j % 500 == 0:
                print(j)
            psd_signal = np.array([])
            features = []
            for sensor_number in range(0, 10):
                #iterate over all sensor's signals
                sensor_id = f'sensor_{sensor_number + 1}'
                element,psd = self.build_features_signal(signals[sensor_id], seg, sensor_id)
                features.append(element)
                psd_signal = np.concatenate((psd_signal, psd), axis=-1)


            features = pd.concat(features,axis=1)
            psd_samples.append(psd_signal)
            self.feature_set.append(features)

            j += 1

        '''
        Dimension reducation for the PSD result
        '''
        psd_samples = np.nan_to_num(np.array(psd_samples))

        if self.__name__ == 'test':
            psd_pca_scaled = self.psd_pca_scaler.transform(psd_samples)
            psd_pca_scaled = np.nan_to_num(psd_pca_scaled)
            psd_after_dimension_reduction = self.psd_pca_transformer.transform(psd_pca_scaled)
        else:
            psd_pca_scaled = self.psd_pca_scaler.fit(psd_samples).transform(psd_samples)
            psd_pca_scaled = np.nan_to_num(psd_pca_scaled)
            psd_after_dimension_reduction = self.psd_pca_transformer.fit_transform(psd_pca_scaled)

        for psd_index in range(psd_after_dimension_reduction.shape[0]):
            for pca_feature_index in range(psd_after_dimension_reduction.shape[1]):
                self.feature_set[psd_index].loc[self.feature_set[psd_index].index.values[0],f'psd_{pca_feature_index}'] = psd_after_dimension_reduction[psd_index][pca_feature_index]

        self.feature_set = pd.concat(self.feature_set)
        self.feature_set = self.feature_set.reset_index()
        self.feature_set = self.feature_set.rename(columns={'index': 'segment_id'})

        self.feature_set = pd.merge(self.feature_set, self.merged, on='segment_id')

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.feature_set.to_csv(f'{self.path}/{self.config.feature_version}_fft_and_psd_stft_{self.__name__}_with_redudant_feat.csv')

    def divide_input_output(self,drop_cols=None):
        if self.__name__ == 'train':
            X = self.feature_set.drop(['segment_id', 'time_to_eruption'], axis=1)
            y = self.feature_set['time_to_eruption']
            return X, y, drop_cols
        elif self.__name__ == 'test':
            X = self.feature_set.drop(['segment_id', 'time_to_eruption'], axis=1)
            return X, self.feature_set['segment_id']
        else:
            raise Exception('Invalid class name please use train or test or define a new name!')

    def remove_redudant_features(self, drop_cols=None):
        if drop_cols is None:
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

        # Drop corresponding columns
        self.feature_set = self.feature_set.drop(drop_cols, axis=1)


        self.feature_set.to_csv(f'{self.path}/{self.config.feature_version}_fft_and_psd_stft_{self.__name__}_without_redudant_feat.csv')

        return self.divide_input_output(drop_cols)

    def load_data_features_before_removing_features(self):
        if not os.path.exists(self.path):
            raise Exception('Missing features! please extract them first')

        self.feature_set = pd.read_csv(f'{self.path}/{self.config.feature_version}_fft_and_psd_stft_{self.__name__}_with_redudant_feat.csv')

        return self.divide_input_output()

    def load_data_features_after_removing_features(self):
        if not os.path.exists(self.path):
            raise Exception('Missing features! please extract them first')

        self.feature_set = pd.read_csv(f'{self.path}/{self.config.feature_version}_fft_and_psd_stft_{self.__name__}_without_redudant_feat.csv')
        return self.divide_input_output()

