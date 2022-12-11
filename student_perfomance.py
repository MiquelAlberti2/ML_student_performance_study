import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import Linear_Regression
from neural_network import Network_Manager

# loading files
df1 = pd.read_csv("dataset2/student-mat.csv", delimiter=";")
df2 = pd.read_csv("dataset2/student-por.csv", delimiter=";")

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

#print("\nGenre types and number of samples for each type:")
print(df['Mjob'].value_counts())

###############
# CLEAN DATASET
###############

col_to_del = ['Mjob', 'Fjob', 'school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery', 'Dalc', 'Walc', 'G1', 'G2']

df.drop(columns=col_to_del, inplace = True, axis=1)

df['sex'] = df['sex'].map({'F':0,
                            'M':1,
                          }).astype(int)

df['address'] = df['address'].map({'U':0,
                                   'R':1,
                                  }).astype(int)

df['famsize'] = df['famsize'].map({'LE3':0,  #less or equal than 3
                                   'GT3':1,  #greater than 3
                                  }).astype(int)

df['Pstatus'] = df['Pstatus'].map({'T':0,
                                   'A':1,
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

df['higher'] = df['higher'].map({'no':0,
                                   'yes':1,
                                  }).astype(int)


df['internet'] = df['internet'].map({'no':0,
                                   'yes':1,
                                  }).astype(int)

df['paid'] = df['paid'].map({'no':0,
                                   'yes':1,
                                  }).astype(int)

df['activities'] = df['activities'].map({'no':0,
                                   'yes':1,
                                  }).astype(int)

df['romantic'] = df['romantic'].map({'no':0,
                                   'yes':1,
                                  }).astype(int)


###############
# LINEAR REGRESSION
###############

numpy_arr = df.to_numpy()
x_list = numpy_arr[:,:-1]
y_list = numpy_arr[:,-1]

print('\nNumpy Array\n----------\n', numpy_arr, len(numpy_arr), len(numpy_arr[0]))

l = Linear_Regression(x_list, y_list)
l.plot_results()

print(l.predictionError(x_list, y_list))

###############
# NEURAL NETWORK
###############
num_classes = 6
num_features = 300

structure = [num_features, 64, num_classes]

model = Network_Manager(structure)