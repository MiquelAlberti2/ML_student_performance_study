import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score


class Linear_Regression():

        def __init__(self, X_train, y_train, reg_params):
            #let's do k-fold cross validation to choose the best parameter
            k=5

            size = int(len(X_train)/k) 
            partitions_X, partitions_y = [0]*k, [0]*k

            for i in range(k):
                partitions_X[i] = X_train[size*i:size*(i+1)]
                partitions_y[i] = y_train[size*i:size*(i+1)]


            best_param = None
            best_error = 10000

            error_param = [0]*len(reg_params)

            param_index = 0
            #decide the best parameter between the ones in the list
            for param in reg_params:
                for i in range(k): # iterate through my partitions
                    X_val = partitions_X[i]
                    y_val = partitions_y[i]

                    first_iter = 0 : ()?
                    for j in range(k):
                        
                        if count == 0:
                            aux_x = (partitions_X[j])
                            aux_y = (partitions_y[j])
                        if j!=i:
                            aux_x+= (partitions_X[j])
                            aux_y+= (partitions_y[j])

                        count +=1

                    self.X_train = np.concatenate(aux_x, axis=0)
                    self.y_train = np.concatenate(aux_y, axis=0)


                    self.theta = self.closedFormSolution(param)
                    error_param[param_index] += self.predictionError(X_val, y_val)
               

                if error_param[param_index] < best_error:
                    best_error = error_param[param_index]
                    best_param = reg_params[param_index]

                param_index += 1

            # Finally, we run regression on the entire dataset using the best param
            self.X_train = X_train
            self.y_train = y_train
            self.theta = self.closedFormSolution(best_param)
  

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

        def closedFormSolution(self, reg_param):
            """
            INPUT: list of lists,
            where the last element of the inner lists is the expected output
            """
            print("computing regression")
            yMatrix = self.y_train
            # add a column of 1s
            phiMatrix = np.insert(self.X_train, len(self.X_train[0]), values=1, axis=1)

            phiT = np.transpose(phiMatrix)
            product = np.matmul(phiT, phiMatrix)
            # add regularization term
            reg_product = np.add(product , reg_param*np.identity(len(product), dtype=int))

            inverse = np.linalg.inv(reg_product)
            product2 = np.matmul(inverse, phiT)

            return np.matmul(product2, yMatrix)

        def plot_results(self):
            print('Result with all data:')
            print('theta: ', self.theta)
            #print('Resulted line: ', self.theta[0], 'x + ', self.theta[1], '...\n')
