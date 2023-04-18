import zipfile
import gdown
from pyparsing import col
import torchcde as cde
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import code_folder.train as train

#data analyze tools

from sklearn.preprocessing import normalize

def get_svd(array):
    dim = 3 
    tmp = np.array(array.x.values)[700: 2500]
    tmp = np.vstack([tmp[i: i + 300] for i in range(tmp.shape[0] - 300)])
    P, D, Q = np.linalg.svd(tmp)
    print(P.shape, D.shape, Q.shape)
    hid = P[:, : dim] @ np.diag(D[:dim])
    # val = P[:, : dim] @ np.diag(D[:dim]) @ Q[:dim]
    return hid  

def stats(array, step):
    means = np.zeros((step, array.shape[1]))
    disps = np.zeros((step, array.shape[1]))
    array = normalize(array, axis = 1)
    for i in range(step):
        neigh = np.array(sorted(array, key = lambda x: np.linalg.norm(array[i] - x)))[:20]
        mean = np.mean(neigh, axis = 0)
        # # print(mean.shape, neigh.shape)
        # print((neigh - mean) ** 2)
        disp = np.mean((neigh - mean) ** 2, axis = 0)
        means[i] = mean
        disps[i] = disp
    return means, disps

def stats_periodic(array, step):
    means = np.zeros((step, array.shape[1]))
    disps = np.zeros((step, array.shape[1]))
    for i in range(step):
        dots = []
        for j in range(i, len(array), step):
            dots.append(array[j])
        dots = np.vstack(dots)
        mean = np.mean(dots, axis = 0)
        disp = np.mean((dots- mean) ** 2, axis = 0)
        means[i] = mean
        disps[i] = disp
    return means, disps

def plots(arrays, markers, to = 1000, x = 40 , y=40):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for arr, mark in zip(arrays, markers):
        tmp = normalize(arr, axis = 1)
        if mark is None:
            ax.plot(tmp[:, 0][:to], tmp[:, 1][:to], tmp[:, 2][:to], label='parametric curve')
        else:
            ax.plot(tmp[:, 0][:to], tmp[:, 1][:to], tmp[:, 2][:to], mark)
    ax.view_init(x, y)

#SSA method
class SSA:
    @staticmethod
    def average_adiag(x):
        x1d = [np.mean(x[::-1, :].diagonal(i)) for i in
           range(-x.shape[0] + 1, x.shape[1])]
        return np.array(x1d)

    @staticmethod
    def SSA(array, emb_len, num_groups, count_in_group = 3):
        # array is 1 dimentional select num_groups * count_in_group biggest singular values
        embedded = np.vstack([array[i :  i + emb_len] for i in range(array.shape[0] - emb_len) ])
        P, D, Q = np.linalg.svd(embedded)
        assert  num_groups * count_in_group  <= len(D)
        print(D[:10])
        groups = []
        for i in range(0, num_groups * count_in_group, count_in_group):
            tmp = (P[:, i: i + count_in_group] * D[i: i + count_in_group])
            # print(P.shape, D.shape, Q.shape, embedded.shape)
            # print(tmp.shape, Q[..., i: i + count_in_group].shape)
            tmp = tmp @ Q[i: i + count_in_group]
            # print(tmp.shape)
            tmp = SSA.average_adiag(tmp)
            # print(tmp.shape)
            # break
            groups.append(tmp)
        
        return groups

    @staticmethod
    def apply_SSA(array, emb_len, count_in_group):
        # возвращает результат применения гусеницы отдельно к осям
        rez = []
        for i in range(array.shape[1]):
            tmp = SSA.SSA(array[..., i], emb_len, 1 , count_in_group)
            rez.append(tmp[0])
        return np.vstack(rez)



# dataPreprocessing tools
class DataPreprocess:
    """
    there all functions which preprocess data
    """    
    @staticmethod
    def get_interpolation(data, time_axis, data_axis, linear = True):
        """
        Линейная интерполяция данных.
        Нужно для выравнивания времен измерения
        """
        data_times = torch.tensor(np.array(data[time_axis]))
        data_linear = torch.tensor(np.array(data[data_axis]))
        if linear:
            data_coeffs = cde.linear_interpolation_coeffs(data_linear, data_times)
            data_interploation = cde.LinearInterpolation(data_coeffs, data_times)
        else:  # else we use qubic interpolation
            data_coeffs = cde.natural_cubic_spline_coeffs(data_linear, data_times)
            data_interploation = cde.CubicSpline(data_coeffs, data_times)
        return data_interploation
    
    @staticmethod
    def align_by_time(X_data, Y_data,  time_axis, data_axis, t = None, linear = True):
        all_axis = [time_axis] + data_axis
        if not (set(all_axis) <= set(X_data.columns) and set(all_axis) <= set(X_data.columns) ): raise ValueError("time_axis and data axis must be in data axis")
        
        X_interpolation = DataPreprocess.get_interpolation(X_data, time_axis, data_axis, linear)
        Y_interpolation = DataPreprocess.get_interpolation(Y_data, time_axis, data_axis, linear)
        if t is None:
            t = X_interpolation.grid_points
        align_x = torch.vstack([X_interpolation.evaluate(t_i) for t_i in t])
        align_y = torch.vstack([Y_interpolation.evaluate(t_i) for t_i in t])

        align_x = torch.hstack([t.reshape(-1, 1), align_x]).numpy()
        align_y = torch.hstack([t.reshape(-1, 1), align_y]).numpy()

        print(align_x.shape)
        align_x = pd.DataFrame(align_x, columns = [ time_axis, *data_axis])
        align_y = pd.DataFrame(align_y, columns = [ time_axis, *data_axis])
        return align_x, align_y

    @staticmethod
    def time_to_delta_t(data, time_axis):
        if time_axis not in data.columns: raise ValueError(f"{time_axis} not in data.columns")
        data[time_axis].iloc[:-1] = np.array(data[time_axis])[1:] - np.array(data[time_axis])[:-1]
        return data.iloc[:-1]


    @staticmethod
    def normalize(data, axis):
        if not (set(axis) <= data.columns): raise ValueError("axis must be in")
        normalizer = StandardScaler()
        data[axis] = normalizer.fit_transform(data[axis])
        return data, normalizer
    
    @staticmethod
    def train_test_split(X, y, t, train_ratio = 0.75):
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t,
            train_size=train_ratio,
            shuffle=False
            )
        return X_train, X_test, y_train, y_test, t_train, t_test


class DatasetReady(Dataset):
    def __init__(self, X, t, embed_dim = -1):
        self.X = torch.tensor( np.array(X))
        self.t = torch.tensor( np.array(t))
        if embed_dim < 0: embed_dim = X.shape[0] - 1
        self.embed_dim = embed_dim
        self.len = X.shape[0] - embed_dim
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.X[index: index + self.embed_dim].T, self.X[index + self.embed_dim].T, self.t[index: index + self.embed_dim]

def get_datasets_pair(X_data, Y_data, emb_dim, data_axis, train_ratio = 0.75, batch_size = 200, shuffle = True):
    # функция для работы с данными акселерометра и гироскопа
    time_axis = "seconds_elapsed"
    if set(X_data.columns) != set(Y_data.columns): raise ValueError("Жесть")
    if not (set(data_axis) <= set(X_data.columns)): raise ValueError("Жесть")
    
    preprocessor = DataPreprocess()
    X_data, Y_data = preprocessor.align_by_time(X_data, Y_data, time_axis, data_axis)
    X_data, Y_data = preprocessor.time_to_delta_t(X_data, time_axis), preprocessor.time_to_delta_t(Y_data, time_axis)
    data_len = X_data.shape[0]
    X_train, X_test = X_data[: int(data_len * train_ratio) ], X_data[int(data_len * train_ratio):]
    Y_train, Y_test = Y_data[: int(data_len * train_ratio) ], Y_data[int(data_len * train_ratio):]
    
    X_train, X_test = DatasetReady(X_train[data_axis], X_train[time_axis], emb_dim) , DatasetReady(X_test[data_axis], X_test[time_axis], emb_dim)
    Y_train, Y_test = DatasetReady(Y_train[data_axis], Y_train[time_axis], emb_dim) , DatasetReady(Y_test[data_axis], Y_test[time_axis], emb_dim)

    X_train, X_test = DataLoader(X_train, batch_size, shuffle),  DataLoader(X_test, batch_size, shuffle)
    Y_train, Y_test =  DataLoader(Y_train, batch_size, shuffle),  DataLoader(Y_test, batch_size, shuffle)

    return X_train, X_test, Y_train, Y_test 

