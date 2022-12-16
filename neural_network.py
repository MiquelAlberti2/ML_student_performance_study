from keras.models import Sequential
from keras.layers import Dense


class Neural_Network:
    def __init__(self, X_train, y_train, setting):
        # Make NN
        self.NN_model = Sequential()

        # Input layer of size X_train.shape[1] (number of features)
        # the first hidden layer has 64 neurons, 
        self.NN_model.add(Dense(setting[0], kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'))

        if len(setting)>1:
            self.NN_model.add(Dense(setting[1], kernel_initializer='normal', activation='relu'))

        # Output Layer, with one node
        self.NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compile Network
        self.NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        self.NN_model.summary()

        self.NN_model.fit(X_train, y_train, epochs=500, verbose=0, batch_size=32, validation_split=0.2)


    def get_acc(self, X, y):
        # Evaluate the model on the test data using `evaluate`
        results, acc = self.NN_model.evaluate(X, y)
        # other way of computing mean error per prediciton
        """predictions = self.NN_model.predict(X_test).flatten()
        differences = np.subtract(predictions, y_test)

        error = 0
        for e in differences:
            error += abs(e)

        mean_e = error / len(differences)

        print('Mean error: ', mean_e)"""

        return acc
