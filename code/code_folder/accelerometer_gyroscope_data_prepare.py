""" 
код обработки данныхх акселерометра и гироскопа
"""

import zipfile
import gdown
from pyparsing import col
import torchcde as cde
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import code_folder.train as train

class DataPreprocess:
    def __init__(self, accelerometer_filename = None, gyroscope_filename = None, data = None,):
        if data is not None:
            self.data = data
        else:
            self.acc_data = pd.read_csv(accelerometer_filename)
            self.gyr_data = pd.read_csv(gyroscope_filename)
            self.aligned = False
            self.embedded = False
    @staticmethod
    def get_interpolation(data, linear = True):
        """
        Линейная интерполяция данных.
        Нужно для выравнивания времен измерения
        """
        data_times = torch.tensor(np.array(data["seconds_elapsed"]))
        data_linear = torch.tensor(np.array(data[["x", "y", "z"]]))
        if linear:
            data_coeffs = cde.linear_interpolation_coeffs(data_linear, data_times)
            data_interploation = cde.LinearInterpolation(data_coeffs, data_times)
        else:  # else we use qubic interpolation
            data_coeffs = cde.natural_cubic_spline_coeffs(data_linear, data_times)
            data_interploation = cde.CubicSpline(data_coeffs, data_times)
        return data_interploation
    @staticmethod
    def _align_by_time_(X_data, Y_data, t = None, linear = True):
        X_interpolation = DataPreprocess.get_interpolation(X_data, linear)
        Y_interpolation = DataPreprocess.get_interpolation(Y_data, linear)
        if t is None:
            t = X_interpolation.grid_points
        align_x = torch.vstack([X_interpolation.evaluate(t_i) for t_i in t])
        align_y = torch.vstack([Y_interpolation.evaluate(t_i) for t_i in t])
        rez = torch.hstack([t.reshape(-1, 1), align_x, align_y]).numpy()
        rez = pd.DataFrame(rez, columns = ["time", "A:x", "A:y", "A:z", "G:x", "G:y", "G:z"])
        return rez
    
    def align_by_time(self):
        self.data = DataPreprocess._align_by_time_(self.acc_data, self.gyr_data)
        self.columns = ["time", "A:x", "A:y", "A:z", "G:x", "G:y", "G:z"]
        self.aligned = True
        return self

    def normalize(self, axis):
        if not (set(axis) <= set(self.columns)): raise ValueError("axis must be in self.columns")
        self.axis_norm = axis
        self.normalizer = StandardScaler()
        self.data[axis] = self.normalizer.fit_transform(self.data[axis])
        return self
        
    def embed(self, embed_dim = 10):
        self.embed_dim = embed_dim
        self.embedded = True
        if not self.aligned: raise Exception("Сначала нужно вызвать метод align")
        time_shifts = list(range(0 , embed_dim + 1))[::-1]
        tmp = pd.DataFrame()
        for shift in time_shifts:
            tmp[[col + f"-{shift}" for col in self.columns]] = self.data.shift(shift)
        tmp.dropna(inplace = True)
        self.data_embed = tmp
        return self
    def embed_to_delta_t(self):
        if not self.embedded: raise Exception("сначала нужно вызвать embed")
        for i in range(self.embed_dim, 0, -1):
            self.data_embed[f"time-{i}"] = self.data_embed[f"time-{i-1}"] - self.data_embed[f"time-{i}"]
        return self
    def preprocess(self, Emb_dim, axis):
    #  чтобы вызвать все сразу
        return self.align_by_time().normalize(axis).embed(Emb_dim).embed_to_delta_t()

class Accessor:
    @staticmethod
    def get_axis(dataset, axis):
    # shape = (data_sim, embed_dim)
        if not hasattr(dataset, "columns"): raise Exception("preprocess dataset!")
        if not(set(axis) <= set(dataset.columns[1:])): raise Exception(f"axis must be in  {dataset.columns[1:]}")
        rez = []
        times = []
        for i in range(dataset.data_embed.shape[0]):
            dat = dataset.data_embed.iloc[i]
            cols = [ax + f"-{i}" for i in range(dataset.embed_dim, -1, -1) for ax in axis]
            time_cols = [f"time-{i}" for i in range(dataset.embed_dim, -1, -1)]
            tmp_rez = np.array(dat[cols]).reshape((-1, len(axis))).T[None, :]
            tmp_times = np.array(dat[time_cols]).reshape((1, -1))[None, :]
            rez.append(tmp_rez)
            times.append(tmp_times)
        rez = torch.tensor(  np.concatenate(rez, axis = 0))
        times = torch.tensor( np.concatenate(times, axis = 0))
       
        return rez[..., : -1], rez[..., -1], times[..., : -1]

    @staticmethod
    def train_test_split(X, y, t, train_ratio = 0.75):
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t,
            train_size=train_ratio,
            shuffle=False
            )
        return X_train, X_test, y_train, y_test, t_train, t_test

class DatasetReady(Dataset):
    def __init__(self, X, y, t):
        self.X = X
        self.y = y
        self.t = t
        self.len_ = X.shape[0]
    
    def __len__(self):
        return self.len_
    
    def __getitem__(self, index: int):
        return self.X[index].type(torch.float32), self.y[index].type(torch.float32), self.t[index].type(torch.float32)

class RandomDataset(Dataset):
    def __init__(self, data_dim, time_emb, len_ = 2000):
        self.data_dim = data_dim
        self.time_emb = time_emb
        self.len_ = len_
    def __len__(self):
        return self.len_
    def __getitem__(self, index: int):
        return torch.randn((self.data_dim, self.time_emb)), torch.randn((self.data_dim)), torch.ones((1, self.time_emb)) / 100

def make_datasets(axis, accelerometer_filename, gyroscope_filename,Emb_dim = 10 , train_ratio = 0.75, batch_size = 200, shuffle = True):
    dataset = DataPreprocess(accelerometer_filename, gyroscope_filename)
    dataset.preprocess(Emb_dim, axis)
    tmp = Accessor.get_axis(dataset, axis)
    tmp = Accessor.train_test_split(*tmp, train_ratio)
    train_ = tmp[0], tmp[2], tmp[4]
    test_ = tmp[1], tmp[3], tmp[5]
    train_ = DatasetReady(*train_)
    test_ = DatasetReady(*test_)
    train_ = DataLoader(train_, batch_size, drop_last = True, shuffle = shuffle)
    test_ = DataLoader(test_, batch_size, drop_last = True, shuffle = shuffle)
    return train_, test_

def make_random_dataset(data_dim, Emb_dim = 10, len_ = 2000, train_ratio = 0.75, batch_size = 200, shuffle = True):
    columns = ["time"] + [f"{i}" for i in range(data_dim)]
    data = pd.DataFrame(np.hstack( [np.ones((len_, 1)) * 0.016, np.random.rand(len_, data_dim)] ), columns = columns)
    dataset = DataPreprocess(data = data)
    dataset.columns = columns
    dataset.aligned = True
    dataset.normalize(columns[1:])
    dataset.embed(Emb_dim ) #.embed_to_delta_t()
    tmp = Accessor.get_axis(dataset, columns[1:])
    tmp = Accessor.train_test_split(*tmp, train_ratio)
    train_ = tmp[0], tmp[2], tmp[4]
    test_ = tmp[1], tmp[3], tmp[5]
    train_ = DatasetReady(*train_)
    test_ = DatasetReady(*test_)
    train_ = DataLoader(train_, batch_size, drop_last = True, shuffle = shuffle)
    test_ = DataLoader(test_, batch_size, drop_last = True, shuffle = shuffle)
    return train_, test_
    return tr_dataset, test_dataset


    


    
    