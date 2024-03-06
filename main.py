import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv("penguins.csv")  # Read penguins.csv file
df = pd.DataFrame(df)
df = df[df.sex.notnull()]  # Delete rows without sex

# Change 5 data columns to list
b_len_list = df['bill_length_mm'].tolist()
b_depth_list = df['bill_depth_mm'].tolist()
f_len_list = df['flipper_length_mm'].tolist()
b_mass_list = df['body_mass_g'].tolist()
sex_list = df['sex'].tolist()

# Change sex to 0 and 1 data
Y = []  # Target
for i in range(len(sex_list)):
    if sex_list[i] == 'female':
        Y.append(0)
    elif sex_list[i] == 'male':
        Y.append(1)
Y = np.array(Y)
# print(np.shape(Y))

    # Get largest element of every list
    b_len_list_max = max(b_len_list)
    b_depth_list_max = max(b_depth_list)
    f_len_list_max = max(f_len_list)
    b_mass_list_max = max(b_mass_list)

    # Scale data to range 0 to 1
    b_len = np.array(b_len_list) / b_len_list_max
    b_depth = np.array(b_depth_list) / b_depth_list_max
    f_len = np.array(f_len_list) / f_len_list_max
    b_mass = np.array(b_mass_list) / b_mass_list_max

    # Concatenate four arrays into one array X
    X = np.vstack((b_len, b_depth, f_len, b_mass))
    X = np.transpose(X)

    # print(np.shape(X))
    # print(X)
    # print(np.shape(b_len))
    # print(np.shape(b_depth))
    # print(np.shape(f_len))
    # print(np.shape(b_mass))

class LogitRegression():

    def __init__(self, learning_rate, iteration_num) -> None:
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num
        self.w = np.random.uniform(low=0, high=1, size=(4,))
        # print(weights)
        # print(np.shape(weights))

    def sigmoid(self, z):
        theta = 1 / (1 + np.exp(-z))
        return theta

    def propagate(self, w, b, X, Y):
        n = 333  # Number of samples

        # Forward Propagate
        h = self.sigmoid(np.dot(w.T, X) + b)
        cost = -(np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))) / n

        # Back propagate
        dZ = h - Y
        dw = (np.dot(X, dZ.T)) / n
        db = (np.sum(dZ)) / n

        # Return value
        grads = {'dw': dw,
                 'db': db}

        return grads, cost

    def optimize(self, w, b, X, Y, iteration_num, learning_rate, print_cost=False):
        costs = []
        for i in range(iteration_num):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads['dw']
            db = grads['db']

            # Use gradient to update w and b
            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print('cost after iteration %i: %f' % (i, cost))
        params = {'w': w,
                  'b': b}
        grads = {'dw': dw,
                 'db': db}
        return params, grads, costs

    def predict(self, w, b, X):
        n = 333
        Y_prediction = np.zeros((1, n))

        h = self.sigmoid(np.dot(w.T, X) + b)
        for i in range(n):
            if h[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        return Y_prediction

    def logistic_model(self, X_train, Y_train, X_test, Y_test, learning_rate=0.000001, iteration_nums=10000,
                       print_cost=False):
        dim = X_train.shape[0]
        W = np.random.uniform(low=0, high=1, size=(4,))
        b = 0

        params, grads, costs = self.optimize(W, b, X_train, Y_train, iteration_nums, learning_rate, print_cost)
        W = params['w']
        b = params['b']

        prediction_train = self.predict(W, b, X_test)
        prediction_test = self.predict(W, b, X_train)

        accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
        accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
        print("Accuracy on train set:", accuracy_train)
        print("Accuracy on test set:", accuracy_test)

        d = {"costs": costs,
             "Y_prediction_test": prediction_test,
             "Y_prediction_train": prediction_train,
             "w": W,
             "b": b,
             "learning_rate": learning_rate,
             "iteration_nums": iteration_nums,
             "train_acy": train_acy,
             "test_acy": test_acy
             }
        return d

    d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
                       print_cost=True)