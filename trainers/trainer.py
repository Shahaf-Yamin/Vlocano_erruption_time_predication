import numpy as np
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn.metrics import mutual_info_score

logger = logging.getLogger("logger")


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


def pred_mae(y_true, y_pred):
    return mae(y_true, y_pred)

def classify_according_time2erruption(y_train,y_val,config):
    y_max = np.max(y_train)
    y_min = np.min(y_train)
    divide = np.round((y_max - y_min) / config.knn_num_classes)
    sections = np.arange(start=y_min, stop=y_max, step=divide)

    classifer_array = y_train.copy()
    for i in range(sections.shape[0] - 1):
        lower_bound = sections[i]
        upper_bound = sections[i + 1]

        a1 = lower_bound < classifer_array
        b1 = classifer_array < upper_bound
        classifer_array[np.bitwise_and(a1, b1)] = i

    if y_val is not None:
        classifer_val_array = y_val.copy()
        for i in range(sections.shape[0] - 1):
            lower_bound = sections[i]
            upper_bound = sections[i + 1]
            a1 = lower_bound < classifer_val_array
            b1 = classifer_val_array < upper_bound
            classifer_val_array[np.bitwise_and(a1, b1)] = i
    else:
        classifer_val_array = None

    return classifer_array,classifer_val_array,sections

def generate_predications(model, data, config):
    classifier_array, classifer_val_array, sections = classify_according_time2erruption(data['train'][1], data['val'][1], config)

    classfier_model = KNeighborsClassifier(n_neighbors=config.knn_num_of_neighbors,metric=mutual_info_score)  # , metric = gaussian_kernel)
    classfier_model.fit(data['train'][0], classifier_array)
    predicted = classfier_model.predict(data['test'])
    predicted_train = classfier_model.predict(data['train'][0])

    train_accuracy = np.mean(predicted_train.ravel() == classifier_array.ravel()) * 100
    print(f"train_accuracy: {train_accuracy}")

    if data['val'][0] is not None:
        predicted_val = classfier_model.predict(data['val'][0])
        val_accuracy = np.mean(predicted_val.ravel() == classifer_val_array.ravel()) * 100
        print(f"validation_accuracy: {val_accuracy}")
        final_pred_val = np.zeros_like(predicted_val)

    final_pred = np.zeros_like(predicted)
    final_pred_train = np.zeros_like(predicted_train)

    for i in range(sections.shape[0] - 1):
        indices_train = predicted_train == i
        indices_test = predicted == i

        # Regression
        if indices_test.shape[0] == 0:
            continue

        model['train'].fit(data['train'][0][indices_train, :], data['train'][1].values[indices_train])
        p = model['train'].predict(data['test'][indices_test, :])
        p_train = model['train'].predict(data['train'][0][indices_train, :])
        final_pred_train[indices_train] = p_train
        final_pred[indices_test] = p

        if data['val'][0] is not None:
            indices_val = predicted_val == i

            final_pred_val[indices_val] = model['train'].predict(data['val'][0][indices_val, :])

    p_test = final_pred.copy()
    p_test[p_test < 0] = np.min(data['train'][1])

    y_train_labels = data['train'][1].values
    print(f'MAE train {config.model_name} : {pred_mae(y_train_labels, final_pred_train)}')

    if data['val'][0] is not None:
        y_val_labels = data['val'][1].values
        print(f'MAE validation {config.model_name} : {pred_mae(y_val_labels, final_pred_val)}')
        return p_test, final_pred_train, final_pred_val

    return p_test, final_pred_train, None



def train(model, data, config):
    # Classifier
    # org_num_of_neighbors = config.knn_num_classes
    # class_offset = 1
    # p_test = np.zeros((class_offset*2+1,data['test'].shape[0]))
    # y_train_labels = data['train'][1].values
    # p_train = np.zeros((class_offset*2+1,y_train_labels.shape[0]))
    #
    # if data['val'][1] is not None:
    #     y_val_labels = data['val'][1].values
    #     p_val = np.zeros((class_offset*2+1,y_val_labels.shape[0]))
    #
    # for index,offset in enumerate(range(org_num_of_neighbors-class_offset,org_num_of_neighbors+class_offset+1)):
    #     config.knn_num_of_neighbors = offset
    #     print(50*'=')
    #     print(f'Number of classes is {offset}')
    #     print(50*'=')
    #     if data['val'][1] is not None:
    #         p_test[index],p_train[index],p_val[index] = generate_predications(model, data, config)
    #     else:
    #         p_test[index], p_train[index], trash = generate_predications(model, data, config)
    #
    # print(50 * '=')
    # print(f'Final predications')
    # print(50 * '=')
    # p_test = np.mean(p_test,axis=0)
    # p_train = np.mean(p_train,axis=0)
    # print(f'MAE train {config.model_name} : {pred_mae(y_train_labels, p_train)}')
    #
    # if data['val'][0] is not None:
    #     p_val = np.mean(p_val,axis=0)
    #     print(f'MAE validation {config.model_name} : {pred_mae(y_val_labels, p_val)}')
    org_num_clusses = config.knn_num_classes
    # for cluseter_number in range(15,org_num_clusses):
    #     print(50 * '=')
    #     # print(f'Predications for {cluseter_number}')
    #     print(50 * '=')
        # config.knn_num_classes = cluseter_number
    p_test, final_pred_train, trash = generate_predications(model, data, config)

    return p_test