import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from linear_regression import Linear_Regression
from neural_network import Neural_Network
from stats_analysis import Statistics


def oneHotEncode(df, colNames):
    for col in colNames:
        if (df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


# loading files
df1 = pd.read_csv("datasets/student-mat.csv", delimiter=";")
df2 = pd.read_csv("datasets/student-por.csv", delimiter=";")

df = pd.concat([df1, df2])
###############
# DATASET VISUALIZATION
###############

print("\n------------Reading file------------")
# Read the CSV input file

print("\nGeneral information of the dataset:")
print(df.info())

print("\nFirst 5 rows, to get an idea of how is the dataset:")
print(df.head(5))

print("\nNumber of samples:")
print(len(df))

###############
# CLEAN DATASET
###############

col_to_del = ['school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery', 'Dalc', 'Walc']
df.drop(columns=col_to_del, inplace=True, axis=1)

df = oneHotEncode(df, ['Fjob', 'Mjob'])

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

stat = Statistics(df)
stat.plot_study_time(), stat.show_corr()
features = df.columns.tolist()
features.remove('G3')
print(stat.linear_model_creator(features, "G3"))

df_y = df.pop("G3")

print("Cleaned features: \n", df.head(5))
print("Cleaned y values: \n", df_y.head(5))
print("Final Features:")
print(df.info())

all_x_array = df.to_numpy()
all_y_array = df_y.to_numpy()

# Dividing the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(all_x_array, all_y_array, test_size=0.2)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

###############
# LINEAR REGRESSION
###############
print("\nLINEAR REGRESSION:")

# Regularization terms that we want to try
reg_param = [i / 4 for i in range(15)]

l = Linear_Regression(X_train, y_train, reg_param)
i_big_coef, i_small_coef = l.plot_results()

print(f'Index {i_big_coef} corresponds to the feature {df.columns[i_big_coef]}')
print(f'Index {i_small_coef} corresponds to the feature {df.columns[i_small_coef]}')

print('Mean error committed: ', l.predictionError(X_test, y_test))

###############
# NEURAL NETWORK
###############
print("\nNEURAL NETWORK:")

model = Neural_Network(X_train, y_train)
model.plot_results(X_test, y_test)
