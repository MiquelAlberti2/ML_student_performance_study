import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score


class My_Linear_Regression:
    def __init__(self, X_train, y_train, reg_params):
        # we do k-fold cross validation to choose the best parameter
        k = 5

        size = int(len(X_train) / k)
        # we create k partitions
        partitions_X, partitions_y = [0] * k, [0] * k

        for i in range(k):
            partitions_X[i] = X_train[size * i:size * (i + 1)]
            partitions_y[i] = y_train[size * i:size * (i + 1)]

        best_param = None
        best_error = 10000  # big enough error so that it gets overwritten

        # error associated to each parameter, to choose the best one
        error_param = [0] * len(reg_params)

        # decide the best parameter between the ones in the list
        for param_index in range(len(reg_params)):
            for i in range(k):  # iterate through my partitions
                X_val = partitions_X[i]
                y_val = partitions_y[i]

                # union of all the other partitions
                for j in range(k):
                    if (i != 0 and j == 0) or (i == 0 and j == 1):
                        # first iteration can't be a concatenate
                        aux_x = partitions_X[j]
                        aux_y = partitions_y[j]
                        # when i==0 and j==0 do nothing
                    elif j != i:
                        aux_x = np.concatenate([aux_x, partitions_X[j]], axis=0)
                        aux_y = np.concatenate([aux_y, partitions_y[j]], axis=0)

                self.X_train = aux_x
                self.y_train = aux_y

                # compute the solution with training dataset
                self.theta = self.closedFormSolution(reg_params[param_index])
                # update the total error for the given parameter
                # using the validation dataset
                error_param[param_index] += self.predictionError(X_val, y_val)

            if error_param[param_index] < best_error:
                best_error = error_param[param_index]
                print("new best error", best_error)
                best_param = reg_params[param_index]

        # Finally, we run regression on the entire dataset using the best param
        self.X_train = X_train
        self.y_train = y_train
        self.theta = self.closedFormSolution(best_param)

    def predictiveFunction(self, x):
        # x is a vector of features
        result = 0
        # compute the dot product
        for i in range(len(self.theta) - 1):
            result += x[i] * self.theta[i]

        # add the offset
        result += self.theta[-1]

        if result < 0:
            result = 0
        elif result > 20:
            result = 20

        return result

    def predictionError(self, x_list, y_list):
        error = 0
        for i in range(len(x_list)):
            error += abs(self.predictiveFunction(x_list[i]) - y_list[i])

        return error / len(x_list)

    def closedFormSolution(self, reg_param):
        """reg_param: regularization term"""
        yMatrix = self.y_train
        # add a column of 1s
        phiMatrix = np.insert(self.X_train, len(self.X_train[0]), values=1, axis=1)

        phiT = np.transpose(phiMatrix)
        product = np.matmul(phiT, phiMatrix)
        # add regularization term
        reg_product = np.add(product, reg_param * np.identity(len(product), dtype=int))

        inverse = np.linalg.inv(reg_product)
        product2 = np.matmul(inverse, phiT)

        return np.matmul(product2, yMatrix)

    def plot_results(self):
        print('************** Resulted coefficients of the hyperplane **************')
        print(self.theta)
        biggest_theta = -20
        smallest_theta = 0

        biggest_theta2 = -20 #second highest coef
        smallest_theta2 = 0 #second smallest coef

        index_b = None
        index_b2 = None
        index_s = None
        index_s2 = None

        # we exclude the offset
        for i in range(len(self.theta)-1):
            if self.theta[i] > biggest_theta:
                biggest_theta2 = biggest_theta
                biggest_theta = self.theta[i]
                index_b2 = index_b
                index_b = i
            elif self.theta[i] > biggest_theta2:
                biggest_theta2 = self.theta[i]
                index_b2 = i

            if self.theta[i] < smallest_theta:
                smallest_theta2 = smallest_theta
                smallest_theta = self.theta[i]
                index_s2 = index_s
                index_s = i
            elif self.theta[i] < smallest_theta2:
                smallest_theta2 = self.theta[i]
                index_s2 = i

        print(f'The biggest coefficient is {biggest_theta} which corresponds to the feature {index_b}')
        print(f'The second biggest coefficient is {biggest_theta2} which corresponds to the feature {index_b2}')
        print(f'The smallest coefficient is {smallest_theta} which corresponds to the feature {index_s}')
        print(f'The second smallest coefficient is {smallest_theta2} which corresponds to the feature {index_s2}')


        return index_b, index_b2, index_s, index_s2
