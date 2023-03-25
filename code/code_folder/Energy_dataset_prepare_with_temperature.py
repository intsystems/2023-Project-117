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

df_temp_emb = df.iloc[:,4:5]
df_temp_emb.columns = ['temperature']

for shift in time_shifts:
    df_emb[f'energy-{shift}h'] = df_emb.iloc[:, 0].shift(shift)

for shift in time_shifts:
    df_temp_emb[f"temperature-{shift}h"] = df_temp_emb.iloc[:, 0].shift(shift)

df_emb.dropna(inplace=True)
df_temp_emb.dropna(inplace = True)

X_energy = df_emb.iloc[:, 1:].values
y_energy = df_emb.iloc[:, 0].values

X_temp = df_temp_emb.iloc[:, 1:].values
y_temp = df_temp_emb.iloc[:, 0].values

X = np.concatenate([X_energy[...,None], X_temp[...,  None]], axis = 2)
y = np.concatenate([y_energy[...,None], y_temp[...,  None]], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=train_ratio,
    shuffle=False
)



print('Train size =', X_train.shape[0])
print('Test size =', X_test.shape[0])

## нормализация
energy_scaler = StandardScaler()
target_scaler = StandardScaler()

emb_size = len(time_shifts)
train_size = X_train.shape[0]
test_size =  X_test.shape[0]
X_train_norm = energy_scaler.fit_transform(X_train.reshape((train_size, -1))  ).reshape((train_size, emb_size , -1))
X_test_norm = energy_scaler.transform(X_test.reshape((test_size, -1))  ).reshape((test_size, emb_size , -1))

y_train_norm = target_scaler.fit_transform(y_train).squeeze()
y_test_norm = target_scaler.transform(y_test).squeeze()

# построение датасета
class EnergyDataset(Dataset):
    def __init__(self, energy_temp_data: np.ndarray, target: np.ndarray):
        assert len(energy_temp_data) == len(target)
        self.data = torch.from_numpy(energy_temp_data).permute(0, 2, 1).float()
        self.target = torch.from_numpy(target).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(f'{self.__class__.__name__} index out of range')
            
        return self.data[index], self.target[index]
    

train_dataset = EnergyDataset(X_train_norm, y_train_norm)
test_dataset = EnergyDataset(X_test_norm, y_test_norm)

# Dataloader

batch_size = 200
tr_dataloader = DataLoader(train_dataset, batch_size, drop_last=True)
tst_dataloader = DataLoader(test_dataset, batch_size, drop_last=True)







