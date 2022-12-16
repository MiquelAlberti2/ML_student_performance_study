import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class Statistics:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_study_time(self):
        plt.figure(figsize=(10, 5))

        plt.title("Grades based on study time")
        sns.regplot(x="studytime", y="G3", data=self.df, color="green", label="G3: Final Period Grades",
                    scatter=False, ci=None)
        plt.xlabel("Study Time")
        plt.ylabel("Grades")

        plt.legend(loc="upper left")
        plt.show()

    def plot_failures(self):
        plt.figure(figsize=(10, 5))

        plt.title("Grades based on failures")
        sns.regplot(x="failures", y="G3", data=self.df, color="red", label="G3: Final Period Grades",
                    scatter=False, ci=None)
        plt.xlabel("Failures")
        plt.ylabel("Grades")

        plt.legend(loc="upper right")
        plt.show()

    def show_medu(self):
        grade = self.df['G3']
        sns.barplot(x=self.df['Medu'], y=grade, hue=self.df['sex'], capsize=.2).set(title='Mother Education and '
                                                                                          'Final Grades')
        plt.show()

    def show_studytime(self):
        grade = self.df['G3']
        sns.barplot(x=self.df['studytime'], y=grade, capsize=.2).set(title='Study Time and Final Grades')
        plt.show()

    def show_higher(self):
        grade = self.df['G3']
        sns.barplot(x=self.df['higher'], y=grade, hue=self.df['sex'], capsize=.2).set(title='Higher Education and '
                                                                                            'Final Grades')
        plt.show()

    def show_failures(self):
        grade = self.df['G3']
        sns.barplot(x=grade, y=self.df['failures'], hue=self.df['sex'], capsize=.2).set(title='Failures and '
                                                                                              'Final Grades')
        plt.show()

    def show_corr(self):
        corr = self.df.corr()
        plt.figure(figsize=(15, 8))
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                    cmap="Spectral")  # visualize correlation matrix
        plt.title('Correlation Heatmap')
        plt.show()

    # Function to create Linear Models with Scikit-Learn

    def linear_model_creator(self, feature_variables, target_variable, test_size=0.2):
        # Algorithm

        # Predict variable (X) and target variable (Y)
        X = np.array(self.df[feature_variables])
        Y = np.array(self.df[target_variable])

        # Dividing the data in training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

        # Creating the Linear model
        lm = LinearRegression()
        lm.fit(X_train, Y_train)

        # Creating predictions with the model
        predictions = lm.predict(X_test)

        # Creating the DF with the used features, the target variable and the predicted data
        dictionary = {}

        for feature in feature_variables:
            dictionary[feature] = X_test[:, feature_variables.index(feature)]

        lm_results = pd.DataFrame.from_dict(dictionary)
        lm_results["Y_target"] = Y_test
        lm_results["Y_predicted"] = predictions

        print("Target:", target_variable)
        print("Features:", feature_variables)
        print("The accuracy (R²) of the Model is:", lm.score(X_test, Y_test))
        print("The MAE of the Model is:", mean_absolute_error(Y_test, predictions))
        print("The intercept (alpha) is:", lm.intercept_)
        print("The coeficients (betas) are: ", lm.coef_)

        return lm_results.head()
