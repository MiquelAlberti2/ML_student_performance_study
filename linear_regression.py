import numpy as np
from matplotlib import pyplot as plt


class Linear_Regression():

        def __init__(self, x_tr_list, y_tr_list):
            self.x_tr_list = x_tr_list
            self.y_tr_list = y_tr_list
            self.theta = self.closedFormSolution()

        def predictiveFunction(self,x):
            # x is a vector of features
            result = 0
            for i in range(len(self.theta)-1):
                result += x[i]*self.theta[i]

            result += self.theta[-1]

            if result<0:
                result = 0
            elif result>20:
                result=20

            return result

        def predictionError(self, x_list, y_list):
            error = 0
            for i in range(len(x_list)):
                error += abs(self.predictiveFunction(x_list[i]) - y_list[i])

            return error/len(x_list)

        def closedFormSolution(self):
            """
            INPUT: list of lists,
            where the last element of the inner lists is the expected output
            """
            print("computing regression")
            yMatrix = self.y_tr_list
            # add a column of 1s
            phiMatrix = np.insert(self.x_tr_list, len(self.x_tr_list[0]), values=1, axis=1)

            print(phiMatrix)
            print(yMatrix)

            phiT = np.transpose(phiMatrix)
            product = np.matmul(phiT, phiMatrix)
            inverse = np.linalg.inv(product)
            product2 = np.matmul(inverse, phiT)

            return np.matmul(product2, yMatrix)

        def plot_results(self):
            print('Result with all data:')
            print('theta: ', self.theta)
            #print('Resulted line: ', self.theta[0], 'x + ', self.theta[1], '...\n')