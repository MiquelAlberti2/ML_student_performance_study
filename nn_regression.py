from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

df1 = pd.read_csv("datasets/student-mat.csv", delimiter=";")
df2 = pd.read_csv("datasets/student-por.csv", delimiter=";")

df = pd.concat([df1, df2])

col_to_del = ['Mjob', 'Fjob', 'school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery', 'Dalc', 'Walc', 'G1',
              'G2']

df.drop(columns=col_to_del, inplace=True, axis=1)

df['sex'] = df['sex'].map({'F': 0,
                           'M': 1,
                           }).astype(int)

df['address'] = df['address'].map({'U': 0,
                                   'R': 1,
                                   }).astype(int)

df['famsize'] = df['famsize'].map({'LE3': 0,  # less or equal than 3
                                   'GT3': 1,  # greater than 3
                                   }).astype(int)

df['Pstatus'] = df['Pstatus'].map({'T': 0,
                                   'A': 1,
                                   }).astype(int)

"""df['Mjob'] = df['Mjob'].map({'teacher':1,
                             'health':2,
                             'services':3,
                             'at_home':4,
                             'other':5
                            }).astype(int)

df['Fjob'] = df['Fjob'].map({'teacher':1,
                             'health':2,
                             'services':3,
                             'at_home':4,
                             'other':5
                            }).astype(int)"""

df['higher'] = df['higher'].map({'no': 0,
                                 'yes': 1,
                                 }).astype(int)

df['internet'] = df['internet'].map({'no': 0,
                                     'yes': 1,
                                     }).astype(int)

df['paid'] = df['paid'].map({'no': 0,
                             'yes': 1,
                             }).astype(int)

df['activities'] = df['activities'].map({'no': 0,
                                         'yes': 1,
                                         }).astype(int)

df['romantic'] = df['romantic'].map({'no': 0,
                                     'yes': 1,
                                     }).astype(int)

train, test = train_test_split(df, test_size=0.2)

"""train_numpy = train.to_numpy()
test_numpy = test.to_numpy()

X_train = train_numpy[:, :-1]
y_train = train_numpy[:, -1]
X_test = test_numpy[:, :-1]
y_test = test_numpy[:, -1]"""

# Make NN
NN_model = Sequential()
# Input Layer
NN_model.add(Dense(20, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

# Hidden Layers
NN_model.add(Dense(64, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(64, kernel_initializer='normal', activation='relu'))

# Output Layer
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile Network
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
