import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error 


from linear_regression import My_Linear_Regression
from neural_network import Neural_Network
from pytorch_NN import My_Network
from stats_analysis import Statistics

from scipy.stats import norm
import statistics


def plot_gauss_distribution(values):
    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    mu, std = norm.fit(values)

    # Plot the histogram.
    plt.hist(values, bins=25, density=True, alpha=0.6, color='b')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)

    plt.show()


def oneHotEncode(df, colNames):
    for col in colNames:
        if df[col].dtype == np.dtype('object'):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


# loading files
df1 = pd.read_csv("datasets/student-mat.csv", delimiter=";")
df2 = pd.read_csv("datasets/student-por.csv", delimiter=";")

df = pd.concat([df1, df2])

# plot_gauss_distribution(df['G3'])
# plot_gauss_distribution(df1['G3'])
# plot_gauss_distribution(df2['G3'])

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

col_to_del = ['Fjob', 'Mjob', 'G1', 'G2', 'school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery']

# df = oneHotEncode(df, ['Fjob', 'Mjob'])

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

df_plot = df.copy()
col_to_del_copy = ['Fjob', 'Mjob', 'school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery']
df.drop(columns=col_to_del, inplace=True, axis=1)
df_plot.drop(columns=col_to_del_copy, inplace=True, axis=1)

stat = Statistics(df_plot)
# Plotting study time and failures in relation to final grade
stat.plot_study_time(), stat.plot_failures()
# Plotting the correlation between the features
stat.show_corr()
# Making the linear regression model
features = df.columns.tolist()
features.remove('G3')
print(stat.linear_model_creator(features, "G3"))
# Show how wanting a higher level of edu and mother's education affects a lot
stat.show_higher(), stat.show_medu(), stat.show_studytime()



df_y = df.pop("G3")

print("Cleaned features: \n", df.head(5))
print("Cleaned y values: \n", df_y.head(5))
print("Final Features:")
print(df.info())

all_x_array = df.to_numpy()
all_y_array = df_y.to_numpy()

# Dividing the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(all_x_array, all_y_array, test_size=0.2)

# finally, we normalize our training data
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###############
# LINEAR REGRESSION
###############
print("\nLINEAR REGRESSION:")

# Regularization terms that we want to try
reg_param = [i / 4 for i in range(15)]

l = My_Linear_Regression(X_train, y_train, reg_param)
i_big_coef, i_big_coef2, i_small_coef, i_small_coef2 = l.plot_results()

print(f'Index {i_big_coef} corresponds to the feature {df.columns[i_big_coef]}')
print(f'Index {i_big_coef2} corresponds to the feature {df.columns[i_big_coef2]}')
print(f'Index {i_small_coef} corresponds to the feature {df.columns[i_small_coef]}')
print(f'Index {i_small_coef2} corresponds to the feature {df.columns[i_small_coef2]}')

print('Mean error committed: ', l.predictionError(X_test, y_test))

###############
# LINEAR REGRESSION WITH BASIS FUNCTION EXPANSION
###############
print("\nPOLYNOMIAL REGRESSION:")


poly = PolynomialFeatures(degree = 2)
poly_for_reg = poly.fit_transform(X_train)
poly.fit(poly_for_reg, y_train)

# Use sklearn linear regression as a base
model_reg = LinearRegression()
model_reg.fit(poly_for_reg, y_train)

# prediction of training data
y_pred_tr = model_reg.predict(poly_for_reg)
# prediction of testing data
y_pred_tst = model_reg.predict(poly.fit_transform(X_test))

print("\nError with traingin data: ", mean_absolute_error(y_train, y_pred_tr))
print("Error with testing data: ", mean_absolute_error(y_test, y_pred_tst))


###############
# NEURAL NETWORK
###############
print("\nNEURAL NETWORK:")

# split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# we run the NN for different settings
settings = [[16],[16,16],[32],[32,32],[64],[64,64],[128],[128, 128]]

x_absis = [i for i in range(len(settings))]
acc_tr_list = []
acc_tst_list = []

for setting in settings:

    print('trying structure with hidden layers: ', setting)
    model = Neural_Network(X_train, y_train, setting)

    acc_tr_list.append(model.get_acc(X_train, y_train))
    acc_tst_list.append(model.get_acc(X_test, y_test))

plt.plot(x_absis, acc_tr_list, '-or')
plt.plot(x_absis, acc_tst_list, '-ob')

plt.show()

NUM_OUTPUTS = 1
NUM_FEATURES = 22


# Let's also try our NN developed using pytorch
setting = [NUM_FEATURES, 32, NUM_OUTPUTS]

net = My_Network(setting, X_train, X_val, y_train, y_val)

acc_tst = net.get_accuracy(X_test, y_test)
print('Mean error of the model with testing data: ', acc_tst)
acc_tr = net.get_accuracy(X_train, y_train)
print('Mean error of the model with training data: ', acc_tr)
acc_val = net.get_accuracy(X_val, y_val)
print('Mean error of the model with validation data: ', acc_val)


