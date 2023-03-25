"""
файл для чтения датасета из 
 https://github.com/intsystems/MathematicalForecastingMethods.git
 в папке 'MathematicalForecastingMethods/lab4/vladimirov/EnergyConsumption.xls'
 чтение производится почти как в этом репозитории
только берем эмбеддинги размера 1
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 42  
seed_everything(SEED)

train_ratio = 0.74  # доля выборки которая пойдет в обучение

def print_df_info(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    print(df.describe())
    return df.sample(10)

df = pd.read_excel('MathematicalForecastingMethods/lab4/vladimirov/EnergyConsumption.xls')
print_df_info(df)

df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = df['Date'].dt.hour
df['Month'] = df['Date'].dt.month


# построение погружений
time_shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9][::-1]

df_emb = df.iloc[:, 3:4]
df_emb.columns = ['energy']

for shift in time_shifts:
    df_emb[f'energy-{shift}h'] = df_emb.iloc[:, 0].shift(shift)

df_emb.dropna(inplace=True)
X_energy = df_emb.iloc[:, 1:].values
y_energy = df_emb.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(
    X_energy,
    y_energy,
    train_size=train_ratio,
    shuffle=False
)

print('Train size =', X_train.shape[0])
print('Test size =', X_test.shape[0])

## нормализация
energy_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_norm = energy_scaler.fit_transform(X_train)
X_test_norm = energy_scaler.transform(X_test)

y_train_norm = target_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
y_test_norm = target_scaler.transform(y_test.reshape(-1, 1)).squeeze()

# построение датасета
class EnergyDataset(Dataset):
    def __init__(self, energy_data: np.ndarray, target: np.ndarray):
        assert len(energy_data) == len(target)
        self.energy = torch.from_numpy(energy_data).float()
        self.target = torch.from_numpy(target).float()
    
    def __len__(self):
        return len(self.energy)
    
    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(f'{self.__class__.__name__} index out of range')
            
        return self.energy[index], self.target[index]
    

train_dataset = EnergyDataset(X_train_norm, y_train_norm)
test_dataset = EnergyDataset(X_test_norm, y_test_norm)

# Dataloader

batch_size = 200
tr_dataloader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle = True)
tst_dataloader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle = True)



## строим разреженный датасет

def build_sparse_df(df: pd.DataFrame, alpha: float = 0.33) -> pd.DataFrame:
    assert 0 <= alpha <= 1
    
    n_observations = df.shape[0]
    k_dropped_elems = int(np.round(n_observations*alpha))
    
    ids_to_drop = np.random.randint(0, n_observations, k_dropped_elems)
    
    return df.drop(index=ids_to_drop)

ALPHA = 0.33
sparse_df = build_sparse_df(df, ALPHA).iloc[:, 3:4]
sparse_df.reset_index(inplace=True)
sparse_df.columns = ['hour', 'energy']

# построение эмбеддингов
time_shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9][::-1]
ts_df = sparse_df.iloc[:, 0:1].copy() / 1000

for shift in time_shifts:
    sparse_df[f'energy-{shift}'] = sparse_df.energy.shift(shift)
    ts_df[f'hour-{shift}'] = ts_df.hour.shift(shift)

sparse_df.dropna(inplace=True)
ts_df.dropna(inplace=True)

X_energy = sparse_df.iloc[:, 2:].values
y_energy = sparse_df.iloc[:, 1].values
ts_emb = ts_df.iloc[:, 1:].values

train_ratio = 0.752

X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
    X_energy,
    y_energy,
    ts_emb,
    train_size=train_ratio,
    shuffle=False
)

energy_sparce_scaler = StandardScaler()
target_sparce_scaler = StandardScaler()

X_train_norm = energy_sparce_scaler.fit_transform(X_train)
X_test_norm = energy_sparce_scaler.transform(X_test)

y_train_norm = target_sparce_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
y_test_norm = target_sparce_scaler.transform(y_test.reshape(-1, 1)).squeeze()


batch_size = 150

train_dataset = EnergyDataset(X_train_norm, y_train_norm)
test_dataset = EnergyDataset(X_test_norm, y_test_norm)

tr_dataloader_sparce = DataLoader(train_dataset, batch_size, drop_last=True, shuffle = True)
tst_dataloader_sparce = DataLoader(test_dataset, batch_size, drop_last=True)







