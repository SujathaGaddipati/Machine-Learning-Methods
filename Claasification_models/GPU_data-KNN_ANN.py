import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

# Datasets
GPU_data = pd.read_csv(r"C:\Users\Administrator\Desktop\sgemm_product_dataset\sgemm_product.csv")
GPU_data.columns = [c.replace(' ', '_') for c in GPU_data.columns] # space in col names are replaced by '_'
print(GPU_data.info())

col = GPU_data.loc[ : ,"Run1_(ms)":"Run4_(ms)"]
GPU_data['avg_run'] = col.mean(axis = 1) # calculating avg run from the 4 runs
GPU_data = GPU_data.drop(["Run1_(ms)","Run2_(ms)", "Run3_(ms)", "Run4_(ms)"], axis=1)
GPU_data['avg_run'].hist()
print(GPU_data['avg_run'].describe())

# We will split the dataset in the median value. High speed run time are <= 69.79 (ms) target class '1' and low speed run time are > 69.79 (ms) that is class '0'
GPU_data['Class'] =GPU_data['avg_run'].apply(lambda x: 1 if x <= 69.79 else 0)
GPU_data = GPU_data.drop(["avg_run"], axis = 1)
GPU_data["Class"].value_counts()

# Splitting the dataset into 70% test and 30% train
GPU_train, GPU_test = train_test_split(GPU_data, test_size = 0.30, random_state = 123)
print(GPU_train.shape, GPU_test.shape)

# Setting X and Y variable for the model
GPU_train_X = GPU_train.drop("Class", axis = 1)
GPU_train_Y = GPU_train["Class"]
GPU_test_X = GPU_test.drop("Class", axis = 1)
GPU_test_Y = GPU_test["Class"]
print(GPU_train_X.shape, GPU_train_Y.shape, GPU_test_X.shape, GPU_test_Y.shape)

# ANN implementation for classification
# Different solver experiments - 1
model_1_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_1_1 = model_1_1.fit(GPU_train_X, GPU_train_Y)
predictions_1_1 = model_1_1.predict(GPU_test_X)
print("Confusion matrix for model_1_1 (solver = adam) is: ")
print(confusion_matrix(GPU_test_Y, predictions_1_1))
print(classification_report(predictions_1_1,GPU_test_Y))
print("Accuracy score for the model_1_1 (solver = adam) is: " , accuracy_score(GPU_test_Y, predictions_1_1))
error_1 = 1 - accuracy_score(GPU_test_Y, predictions_1_1)
predict_train = model_1_1.predict(GPU_train_X)
error_train_1 = 1 - accuracy_score(GPU_train_Y, predict_train)


model_1_2 = MLPClassifier(solver = 'sgd', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_1_2 = model_1_2.fit(GPU_train_X, GPU_train_Y)
predictions_1_2 = model_1_2.predict(GPU_test_X)
print("Confusion matrix for model_1_2 (solver = sgd) is: ")
print(confusion_matrix(GPU_test_Y, predictions_1_2))
print(classification_report(predictions_1_2,GPU_test_Y))
print("Accuracy score for the model_1_2 (solver = sgd) is: " , accuracy_score(GPU_test_Y, predictions_1_2))
error_2 = 1 - accuracy_score(GPU_test_Y, predictions_1_2)
predict_train = model_1_2.predict(GPU_train_X)
error_train_2 = 1 - accuracy_score(GPU_train_Y, predict_train)


model_1_3 = MLPClassifier(solver = 'lbfgs', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_1_3 = model_1_3.fit(GPU_train_X, GPU_train_Y)
predictions_1_3 = model_1_3.predict(GPU_test_X)
print("Confusion matrix for model_1_3 (solver = lbfgs) is: ")
print(confusion_matrix(GPU_test_Y, predictions_1_3))
print(classification_report(predictions_1_3,GPU_test_Y))
print("Accuracy score for the model_1_3 (solver = lbgfs) is: " , accuracy_score(GPU_test_Y, predictions_1_3))
error_3 = 1 - accuracy_score(GPU_test_Y, predictions_1_3)
predict_train = model_1_3.predict(GPU_train_X)
error_train_3 = 1 - accuracy_score(GPU_train_Y, predict_train)

# Learning Curve - exp1
test_error = np.array([error_1, error_2, error_3])
train_error = np.array([error_train_1, error_train_2, error_train_3])
KNN_solvers = np.array(['adam', 'sgd', 'lbgfs'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_solvers, test_error, color = 'blue')
plt.plot(KNN_solvers, train_error, color = 'orange')
plt.title("Classification Error with respect to KNN Solvers")
plt.ylabel("Classification Error")
plt.xlabel("KNN_solvers")
plt.legend(["test error", "train_error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# Experiment-2 with activation layer:
model_2_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_2_1 = model_2_1.fit(GPU_train_X, GPU_train_Y)
predictions_2_1 = model_2_1.predict(GPU_test_X)
print("Confusion matrix for model_2_1 with activation as logistic is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_1))
print(classification_report(predictions_2_1,GPU_test_Y))
print("Accuracy score for the model_2_1 with activation as logistic is: " , accuracy_score(GPU_test_Y, predictions_2_1))
error_1 = 1 - accuracy_score(GPU_test_Y, predictions_2_1)
predict_train = model_2_1.predict(GPU_train_X)
error_train_1 = 1 - accuracy_score(GPU_train_Y, predict_train)

model_2_2 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'identity',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_2_2 = model_2_2.fit(GPU_train_X, GPU_train_Y)
predictions_2_2 = model_2_2.predict(GPU_test_X)
print("Confusion matrix for model_2_2 with activation as identity is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_2))
print(classification_report(predictions_2_2,GPU_test_Y))
print("Accuracy score for the model_2_2 with activation as identity is: " , accuracy_score(GPU_test_Y, predictions_2_2))
error_2 = 1 - accuracy_score(GPU_test_Y, predictions_2_2)
predict_train = model_2_2.predict(GPU_train_X)
error_train_2 = 1 -accuracy_score(GPU_train_Y, predict_train)


model_2_3 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_2_3 = model_2_3.fit(GPU_train_X, GPU_train_Y)
predictions_2_3 = model_2_3.predict(GPU_test_X)
print("Confusion matrix for model_2_3 with activation as tanh is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_3))
print(classification_report(predictions_2_3,GPU_test_Y))
print("Accuracy score for the model_2_3 with activation as tanh is: ", accuracy_score(GPU_test_Y, predictions_2_3))
error_3 = 1 - accuracy_score(GPU_test_Y, predictions_2_3)
predict_train = model_2_3.predict(GPU_train_X)
error_train_3 = 1 - accuracy_score(GPU_train_Y, predict_train)

model_2_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'relu',
                        learning_rate_init= 0.001, max_iter = 30, tol = 1e-4)
History_2_4 = model_2_4.fit(GPU_train_X, GPU_train_Y)
predictions_2_4 = model_2_4.predict(GPU_test_X)
print("Confusion matrix for model_2_4 with activation as relu is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_4))
print(classification_report(predictions_2_4,GPU_test_Y))
print("Accuracy score for the model_2_4 with activation as relu is: " , accuracy_score(GPU_test_Y, predictions_2_4))
error_4 = 1 - accuracy_score(GPU_test_Y, predictions_2_4)
predict_train = model_2_4.predict(GPU_train_X)
error_train_4 = 1 - accuracy_score(GPU_train_Y, predict_train)

# Learning Curve exp-2
test_error = np.array([error_1, error_2, error_3, error_4])
train_error = np.array([error_train_1, error_train_2, error_train_3, error_train_4])
KNN_solvers = np.array(['logistic', 'identity', 'tanh', 'relu'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_solvers, test_error, color = 'blue')
plt.plot(KNN_solvers, train_error, color = 'orange')
plt.title("Classification Error with respect to KNN activation layers")
plt.ylabel("Classification Error")
plt.xlabel("KNN_activation_layers")
plt.legend(["test error", "train_error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# ANN iteration Experiment - 3 with tanh activation layer
model_3_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 10, tol = 1e-4)
History_3_1 = model_3_1.fit(GPU_train_X, GPU_train_Y)
predictions_3_1 = model_3_1.predict(GPU_test_X)
print("Confusion matrix for model_3_1 with for 10 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_1))
print(classification_report(predictions_3_1,GPU_test_Y))
print("Accuracy score for the model_3_1 with for 10 iterations: " , accuracy_score(GPU_test_Y, predictions_3_1))
test_acc_1 = 1 - accuracy_score(GPU_test_Y, predictions_3_1)
predict_train = model_3_1.predict(GPU_train_X)
error_train_1 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_2 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 50, tol = 1e-4)
History_3_2 = model_3_2.fit(GPU_train_X, GPU_train_Y)
predictions_3_2 = model_3_2.predict(GPU_test_X)
print("Confusion matrix for model_3_2 with for 20 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_2))
print(classification_report(predictions_3_2,GPU_test_Y))
print("Accuracy score for the model_3_2 with for 20 iterations: " , accuracy_score(GPU_test_Y, predictions_3_2))
test_acc_2 = 1 - accuracy_score(GPU_test_Y, predictions_3_2)
predict_train = model_3_2.predict(GPU_train_X)
error_train_2 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_3 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 70, tol = 1e-4)
History_3_3 = model_3_3.fit(GPU_train_X, GPU_train_Y)
predictions_3_3 = model_3_3.predict(GPU_test_X)
print("Confusion matrix for model_3_2 with for 40 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_3))
print(classification_report(predictions_3_3,GPU_test_Y))
print("Accuracy score for the model_3_3 with for 40 iterations: " , accuracy_score(GPU_test_Y, predictions_3_3))
test_acc_3 = 1 - accuracy_score(GPU_test_Y, predictions_3_3)
predict_train = model_3_3.predict(GPU_train_X)
error_train_3 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 100, tol = 1e-4)
History_3_4 = model_3_4.fit(GPU_train_X, GPU_train_Y)
predictions_3_4 = model_3_4.predict(GPU_test_X)
print("Confusion matrix for model_3_4 with for 100 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_4))
print(classification_report(predictions_3_4,GPU_test_Y))
print("Accuracy score for the model_3_4 with for 100 iterations: " , accuracy_score(GPU_test_Y, predictions_3_4))
test_acc_4 = 1 - accuracy_score(GPU_test_Y, predictions_3_4)
predict_train = model_3_4.predict(GPU_train_X)
error_train_4 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_5 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 150, tol = 1e-4)
History_3_5 = model_3_5.fit(GPU_train_X, GPU_train_Y)
predictions_3_5 = model_3_5.predict(GPU_test_X)
print("Confusion matrix for model_3_5 with for 150 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_5))
print(classification_report(predictions_3_5,GPU_test_Y))
print("Accuracy score for the model_3_5 with for 150 iterations: " , accuracy_score(GPU_test_Y, predictions_3_5))
test_acc_5 = 1 - accuracy_score(GPU_test_Y, predictions_3_5)
predict_train = model_3_5.predict(GPU_train_X)
error_train_5 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_6 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'tanh',
                        learning_rate_init= 0.001, max_iter = 200, tol = 1e-4)
History_3_6 = model_3_6.fit(GPU_train_X, GPU_train_Y)
predictions_3_6 = model_3_6.predict(GPU_test_X)
print("Confusion matrix for model_3_6 with for 200 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_6))
print(classification_report(predictions_3_6,GPU_test_Y))
print("Accuracy score for the model_3_6 with for 200 iterations: " , accuracy_score(GPU_test_Y, predictions_3_6))
test_acc_6 = 1 - accuracy_score(GPU_test_Y, predictions_3_6)
predict_train = model_3_6.predict(GPU_train_X)
error_train_6 = 1 -accuracy_score(GPU_train_Y, predict_train)

# Learning Curve - exp 3
test_error = np.array([test_acc_1, test_acc_2, test_acc_3, test_acc_4, test_acc_5, test_acc_6])
train_error = np.array([error_train_1, error_train_2, error_train_3, error_train_4, error_train_5, error_train_6])
learning_rate = np.array([10, 20, 40, 100, 150, 200])
plt.style.use('seaborn-whitegrid')
plt.plot(learning_rate, test_error, color = 'blue')
plt.plot(learning_rate, train_error, color = 'orange')
plt.title("Classification Error with respect to iteration for tanh")
plt.ylabel("Classification Error")
plt.xlabel("# iterations")
plt.legend(["test error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# ANN iteration Experiment - 3 with logistic activation layer
model_3_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 10, tol = 1e-4)
History_3_1 = model_3_1.fit(GPU_train_X, GPU_train_Y)
predictions_3_1 = model_3_1.predict(GPU_test_X)
print("Confusion matrix for model_3_1 with for 10 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_1))
print(classification_report(predictions_3_1,GPU_test_Y))
print("Accuracy score for the model_3_1 with for 10 iterations: " , accuracy_score(GPU_test_Y, predictions_3_1))
test_acc_1 = 1 - accuracy_score(GPU_test_Y, predictions_3_1)
predict_train = model_3_1.predict(GPU_train_X)
error_train_1 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_2 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 50, tol = 1e-4)
History_3_2 = model_3_2.fit(GPU_train_X, GPU_train_Y)
predictions_3_2 = model_3_2.predict(GPU_test_X)
print("Confusion matrix for model_3_2 with for 20 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_2))
print(classification_report(predictions_3_2,GPU_test_Y))
print("Accuracy score for the model_3_2 with for 20 iterations: " , accuracy_score(GPU_test_Y, predictions_3_2))
test_acc_2 = 1 - accuracy_score(GPU_test_Y, predictions_3_2)
predict_train = model_3_2.predict(GPU_train_X)
error_train_2 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_3 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 70, tol = 1e-4)
History_3_3 = model_3_3.fit(GPU_train_X, GPU_train_Y)
predictions_3_3 = model_3_3.predict(GPU_test_X)
print("Confusion matrix for model_3_2 with for 40 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_3))
print(classification_report(predictions_3_3,GPU_test_Y))
print("Accuracy score for the model_3_3 with for 40 iterations: " , accuracy_score(GPU_test_Y, predictions_3_3))
test_acc_3 = 1 - accuracy_score(GPU_test_Y, predictions_3_3)
predict_train = model_3_3.predict(GPU_train_X)
error_train_3 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 100, tol = 1e-4)
History_3_4 = model_3_4.fit(GPU_train_X, GPU_train_Y)
predictions_3_4 = model_3_4.predict(GPU_test_X)
print("Confusion matrix for model_3_4 with for 100 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_4))
print(classification_report(predictions_3_4,GPU_test_Y))
print("Accuracy score for the model_3_4 with for 100 iterations: " , accuracy_score(GPU_test_Y, predictions_3_4))
test_acc_4 = 1 - accuracy_score(GPU_test_Y, predictions_3_4)
predict_train = model_3_4.predict(GPU_train_X)
error_train_4 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_5 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 150, tol = 1e-4)
History_3_5 = model_3_5.fit(GPU_train_X, GPU_train_Y)
predictions_3_5 = model_3_5.predict(GPU_test_X)
print("Confusion matrix for model_3_5 with for 150 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_5))
print(classification_report(predictions_3_5,GPU_test_Y))
print("Accuracy score for the model_3_5 with for 150 iterations: " , accuracy_score(GPU_test_Y, predictions_3_5))
test_acc_5 = 1 - accuracy_score(GPU_test_Y, predictions_3_5)
predict_train = model_3_5.predict(GPU_train_X)
error_train_5 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_3_6 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 200, tol = 1e-4)
History_3_6 = model_3_6.fit(GPU_train_X, GPU_train_Y)
predictions_3_6 = model_3_6.predict(GPU_test_X)
print("Confusion matrix for model_3_6 with for 200 iteration: ")
print(confusion_matrix(GPU_test_Y, predictions_3_6))
print(classification_report(predictions_3_6,GPU_test_Y))
print("Accuracy score for the model_3_6 with for 200 iterations: " , accuracy_score(GPU_test_Y, predictions_3_6))
test_acc_6 = 1 - accuracy_score(GPU_test_Y, predictions_3_6)
predict_train = model_3_6.predict(GPU_train_X)
error_train_6 = 1 -accuracy_score(GPU_train_Y, predict_train)

# Learning Curve - exp 3
test_error = np.array([test_acc_1, test_acc_2, test_acc_3, test_acc_4, test_acc_5, test_acc_6])
train_error = np.array([error_train_1, error_train_2, error_train_3, error_train_4, error_train_5, error_train_6])
learning_rate = np.array([10, 20, 40, 100, 150, 200])
plt.style.use('seaborn-whitegrid')
plt.plot(learning_rate, test_error, color = 'blue')
plt.plot(learning_rate, train_error, color = 'orange')
plt.title("Classification Error with respect to iteration for logistic")
plt.ylabel("Classification Error")
plt.xlabel("# iterations")
plt.legend(["test error"], loc = 'upper right')
plt.figure(figsize = (5,5))

#ANN Experiments-4 with learning rate
model_4_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.1, max_iter = 20, tol = 1e-4)
History_4_1 = model_4_1.fit(GPU_train_X, GPU_train_Y)
predictions_4_1_test = model_4_1.predict(GPU_test_X)
predictions_4_1_train = model_4_1.predict(GPU_train_X)
print("Confusion matrix for model_4_1 for learning rate = 0.1: ")
print(confusion_matrix(GPU_test_Y, predictions_4_1_test))
print(classification_report(predictions_4_1_test,GPU_test_Y))
test_acc_1 = 1 - accuracy_score(GPU_test_Y, predictions_4_1_test)
train_acc_1 = 1 -  accuracy_score(GPU_train_Y, predictions_4_1_train)
print("Accuracy score for the model_4_1 for learning rate = 0.1: " , accuracy_score(GPU_test_Y, predictions_4_1_test))

model_4_2 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_4_2_test = model_4_2.fit(GPU_train_X, GPU_train_Y)
predictions_4_2_test = model_4_2.predict(GPU_test_X)
print("Confusion matrix for model_4_2 for learning rate = 0.05: ")
print(confusion_matrix(GPU_test_Y, predictions_4_2_test))
print(classification_report(predictions_4_2_test,GPU_test_Y))
print("Accuracy score for the model_4_2 for learning rate = 0.05: " , accuracy_score(GPU_test_Y, predictions_4_2_test))
predictions_4_2_train = model_4_2.predict(GPU_train_X)
test_acc_2 = 1 - accuracy_score(GPU_test_Y, predictions_4_2_test)
train_acc_2 = 1 -  accuracy_score(GPU_train_Y, predictions_4_2_train)


model_4_3 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.001, max_iter = 20, tol = 1e-4)
History_4_3 = model_4_3.fit(GPU_train_X, GPU_train_Y)
predictions_4_3_test = model_4_3.predict(GPU_test_X)
print("Confusion matrix for model_4_3 for learning rate = 0.01: ")
print(confusion_matrix(GPU_test_Y, predictions_4_3_test))
print(classification_report(predictions_4_3_test,GPU_test_Y))
print("Accuracy score for the model_4_3 for learning rate = 0.01: " , accuracy_score(GPU_test_Y, predictions_4_3_test))
predictions_4_3_train = model_4_3.predict(GPU_train_X)
test_acc_3 = 1 - accuracy_score(GPU_test_Y, predictions_4_3_test)
train_acc_3 = 1 - accuracy_score(GPU_train_Y, predictions_4_3_train)


model_4_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.0001, max_iter = 20, tol = 1e-4)
History_4_4 = model_4_4.fit(GPU_train_X, GPU_train_Y)
predictions_4_4_test = model_4_4.predict(GPU_test_X)
print("Confusion matrix for model_4_4 for learning rate = 0.001: ")
print(confusion_matrix(GPU_test_Y, predictions_4_4_test))
print(classification_report(predictions_4_4_test,GPU_test_Y))
print("Accuracy score for the model_4_4 for learning rate = 0.001: " , accuracy_score(GPU_test_Y, predictions_4_4_test))
predictions_4_4_train = model_4_4.predict(GPU_train_X)
test_acc_4 = 1 - accuracy_score(GPU_test_Y, predictions_4_4_test)
train_acc_4 = 1 - accuracy_score(GPU_train_Y, predictions_4_4_train)



model_4_5 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.00001, max_iter = 20, tol = 1e-4)
History_4_5 = model_4_5.fit(GPU_train_X, GPU_train_Y)
predictions_4_5_test = model_4_5.predict(GPU_test_X)
print("Confusion matrix for model_4_5 for learning rate = 0.0001: ")
print(confusion_matrix(GPU_test_Y, predictions_4_5_test))
print(classification_report(predictions_4_5_test,GPU_test_Y))
print("Accuracy score for the model_4_5 for learning rate = 0.0001: " , accuracy_score(GPU_test_Y, predictions_4_5_test))
predictions_4_5_train = model_4_5.predict(GPU_train_X)
test_acc_5 = 1 - accuracy_score(GPU_test_Y, predictions_4_5_test)
train_acc_5 = 1 - accuracy_score(GPU_train_Y, predictions_4_5_train)


# Learning Curve - exp - 4
train_error = np.array([train_acc_5, train_acc_4, train_acc_3, train_acc_2, train_acc_1])
test_error = np.array([test_acc_5, test_acc_4, test_acc_3, test_acc_2, test_acc_1])
learning_rate = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
plt.style.use('seaborn-whitegrid')
plt.plot(learning_rate, train_error, color = 'orange')
plt.plot(learning_rate, test_error, color = 'blue')
plt.title("Classification Error with respect to learning rate")
plt.ylabel("Classification Error")
plt.xlabel("learning rate")
plt.legend(["train error", "test error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# ANN Experiments - Hidden layer
model_5_1 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (5,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_5_1 = model_5_1.fit(GPU_train_X, GPU_train_Y)
predictions_5_1 = model_5_1.predict(GPU_test_X)
print("Confusion matrix for model_5_1: ")
print(confusion_matrix(GPU_test_Y, predictions_5_1))
print(classification_report(predictions_5_1,GPU_test_Y))
print("Accuracy score for the model_5_1 (3,1): " , accuracy_score(GPU_test_Y, predictions_5_1))
error_1 = 1 - accuracy_score(GPU_test_Y, predictions_5_1)
predict_train = model_5_1.predict(GPU_train_X)
error_train_1 = 1 -accuracy_score(GPU_train_Y, predict_train)


model_5_2 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_5_2 = model_5_2.fit(GPU_train_X, GPU_train_Y)
predictions_5_2 = model_5_2.predict(GPU_test_X)
print("Confusion matrix for model_5_2 (2,1): ")
print(confusion_matrix(GPU_test_Y, predictions_5_2))
print(classification_report(predictions_5_2,GPU_test_Y))
print("Accuracy score for the model_5_2 (2,1): " , accuracy_score(GPU_test_Y, predictions_5_2))
error_2 = 1 - accuracy_score(GPU_test_Y, predictions_5_2)
predict_train = model_5_2.predict(GPU_train_X)
error_train_2 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_5_3 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,2), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_5_3 = model_5_3.fit(GPU_train_X, GPU_train_Y)
predictions_5_3 = model_5_3.predict(GPU_test_X)
print("Confusion matrix for model_5_3 (2,2): ")
print(confusion_matrix(GPU_test_Y, predictions_5_3))
print(classification_report(predictions_5_3,GPU_test_Y))
print("Accuracy score for the model_5_3 (2,2): " , accuracy_score(GPU_test_Y, predictions_5_3))
error_3 = 1 - accuracy_score(GPU_test_Y, predictions_5_3)
predict_train = model_5_3.predict(GPU_train_X)
error_train_3 = 1 -accuracy_score(GPU_train_Y, predict_train)

model_5_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (5,2), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_5_4 = model_5_4.fit(GPU_train_X, GPU_train_Y)
predictions_5_4 = model_5_4.predict(GPU_test_X)
print("Confusion matrix for model_5_4 (4,1): ")
print(confusion_matrix(GPU_test_Y, predictions_5_4))
print(classification_report(predictions_5_4,GPU_test_Y))
print("Accuracy score for the model_5_4 (4,1): " , accuracy_score(GPU_test_Y, predictions_5_4))
error_4 = 1 - accuracy_score(GPU_test_Y, predictions_5_4)
predict_train = model_5_4.predict(GPU_train_X)
error_train_4 = 1 -accuracy_score(GPU_train_Y, predict_train)

# Plots
test_error = np.array([error_2, error_3, error_1, error_4])
train_error = np.array([error_train_2, error_train_3, error_train_1, error_train_4])
learning_rate = np.array([3.1, 3.2, 5.1, 5.2])
plt.style.use('seaborn-whitegrid')
plt.plot(learning_rate, test_error, color = 'blue')
plt.plot(learning_rate, train_error, color = 'orange')
plt.title("Classification Error with respect to iteration for logistic")
plt.ylabel("Classification Error")
plt.xlabel("nodes.hidden_layers")
plt.legend(["test error", "train_error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# ANN Experiments-6 with alpha
model_6_1 = MLPClassifier(solver = 'adam', alpha = 0.1, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_6_1 = model_6_1.fit(GPU_train_X, GPU_train_Y)
predictions_6_1_test = model_6_1.predict(GPU_test_X)
predictions_6_1_train = model_6_1.predict(GPU_train_X)
print("Confusion matrix for model_6_1 for alpha = 0.1: ")
print(confusion_matrix(GPU_test_Y, predictions_6_1_test))
print(classification_report(predictions_6_1_test,GPU_test_Y))
test_acc_1 = 1 - accuracy_score(GPU_test_Y, predictions_6_1_test)
train_acc_1 = 1 -  accuracy_score(GPU_train_Y, predictions_6_1_train)
print("Accuracy score for the model_6_1 for alpha = 0.1: " , accuracy_score(GPU_test_Y, predictions_6_1_test))

model_6_2 = MLPClassifier(solver = 'adam', alpha = 0.01, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_6_2 = model_6_2.fit(GPU_train_X, GPU_train_Y)
predictions_6_2_test = model_6_2.predict(GPU_test_X)
predictions_6_2_train = model_6_2.predict(GPU_train_X)
print("Confusion matrix for model_6_2 for alpha = 0.01: ")
print(confusion_matrix(GPU_test_Y, predictions_6_2_test))
print(classification_report(predictions_6_2_test,GPU_test_Y))
test_acc_2 = 1 - accuracy_score(GPU_test_Y, predictions_6_2_test)
train_acc_2 = 1 -  accuracy_score(GPU_train_Y, predictions_6_2_train)
print("Accuracy score for the model_6_2 for alpha = 0.1: " , accuracy_score(GPU_test_Y, predictions_6_2_test))


model_6_3 = MLPClassifier(solver = 'adam', alpha = 0.001, hidden_layer_sizes= (3,1), activation= 'logistic',
                      learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_6_3 = model_6_3.fit(GPU_train_X, GPU_train_Y)
predictions_6_3_test = model_6_3.predict(GPU_test_X)
predictions_6_3_train = model_6_3.predict(GPU_train_X)
print("Confusion matrix for model_6_3 for alpha = 0.001: ")
print(confusion_matrix(GPU_test_Y, predictions_6_3_test))
print(classification_report(predictions_6_3_test,GPU_test_Y))
test_acc_3 = 1 - accuracy_score(GPU_test_Y, predictions_6_3_test)
train_acc_3 = 1 -  accuracy_score(GPU_train_Y, predictions_6_3_train)
print("Accuracy score for the model_6_3 for alpha = 0.001: " , accuracy_score(GPU_test_Y, predictions_6_3_test))



model_6_4 = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_6_4 = model_6_4.fit(GPU_train_X, GPU_train_Y)
predictions_6_4_test = model_6_4.predict(GPU_test_X)
predictions_6_4_train = model_6_4.predict(GPU_train_X)
print("Confusion matrix for model_6_4 for alpha = 0.0001: ")
print(confusion_matrix(GPU_test_Y, predictions_6_4_test))
print(classification_report(predictions_6_4_test,GPU_test_Y))
test_acc_4 = 1 - accuracy_score(GPU_test_Y, predictions_6_4_test)
train_acc_4 = 1 -  accuracy_score(GPU_train_Y, predictions_6_4_train)
print("Accuracy score for the model_6_4 for alpha = 0.0001: " , accuracy_score(GPU_test_Y, predictions_6_4_test))


model_6_5 = MLPClassifier(solver = 'adam', alpha = 0.00001, hidden_layer_sizes= (3,1), activation= 'logistic',
                        learning_rate_init= 0.01, max_iter = 20, tol = 1e-4)
History_6_5 = model_6_5.fit(GPU_train_X, GPU_train_Y)
predictions_6_5_test = model_6_5.predict(GPU_test_X)
print("Confusion matrix for model_6_5 for alpha = 0.00001: ")
print(confusion_matrix(GPU_test_Y, predictions_6_5_test))
print(classification_report(predictions_6_5_test,GPU_test_Y))
print("Accuracy score for the model_6_5 for alpha = 0.00001: " , accuracy_score(GPU_test_Y, predictions_6_5_test))
predictions_6_5_train = model_6_5.predict(GPU_train_X)
test_acc_5 = 1 - accuracy_score(GPU_test_Y, predictions_6_5_test)
train_acc_5 = 1 - accuracy_score(GPU_train_Y, predictions_6_5_train)


# Learning Curve - exp - 4
train_error = np.array([train_acc_5, train_acc_4, train_acc_3, train_acc_2, train_acc_1])
test_error = np.array([test_acc_5, test_acc_4, test_acc_3, test_acc_2, test_acc_1])
alpha = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
plt.style.use('seaborn-whitegrid')
plt.plot(alpha, train_error, color = 'orange')
plt.plot(alpha, test_error, color = 'blue')
plt.title("Classification Error with respect to learning rate")
plt.ylabel("Classification Error")
plt.xlabel("learning rate")
plt.legend(["train error", "test error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# Cross validation experiement
train_sizes, train_scores, test_scores = learning_curve(model_5_2,GPU_train_X,GPU_train_Y, train_sizes= ([0.1,0.33,0.55,0.78,1.]),cv=5)
print("train sizes are: ", train_sizes)
print("train scores are: ", train_scores)
print("test scores are: ", test_scores)

# KNN Experiements -1 (neighbors)
start_time = time.process_time_ns()
model_1 = KNeighborsClassifier(n_neighbors= 1, weights= 'distance', algorithm='auto', p=2)
model_1.fit(GPU_train_X, GPU_train_Y)
predictions_1 = model_1.predict(GPU_test_X)
print("Confusion matrix for model_1 is: ")
print(confusion_matrix(GPU_test_Y, predictions_1))
print(classification_report(predictions_1,GPU_test_Y))
print("Accuracy score for the model_1 is: " , accuracy_score(GPU_test_Y, predictions_1))
Error_1 = 1 -  accuracy_score(GPU_test_Y, predictions_1)
predict_train = model_1.predict(GPU_train_X)
train_error_1 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_1 =  end_time - start_time

start_time = time.process_time_ns()
model_2 = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=2)
model_2.fit(GPU_train_X, GPU_train_Y)
predictions_2 = model_2.predict(GPU_test_X)
print("Confusion matrix for model_2 is: ")
print(confusion_matrix(GPU_test_Y, predictions_2))
print(classification_report(predictions_2,GPU_test_Y))
print("Accuracy score for the model_2 is: " , accuracy_score(GPU_test_Y, predictions_2))
Error_2 = 1 -  accuracy_score(GPU_test_Y, predictions_2)
predict_train = model_2.predict(GPU_train_X)
train_error_2 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_2 =  end_time - start_time

start_time=time.process_time_ns()
model_3 = KNeighborsClassifier(n_neighbors= 5, weights= 'distance', algorithm='auto', p=2)
model_3.fit(GPU_train_X, GPU_train_Y)
predictions_3 = model_3.predict(GPU_test_X)
print("Confusion matrix for model_3 is: ")
print(confusion_matrix(GPU_test_Y, predictions_3))
print(classification_report(predictions_3,GPU_test_Y))
print("Accuracy score for the model_3 is: " , accuracy_score(GPU_test_Y, predictions_3))
Error_3 = 1 -  accuracy_score(GPU_test_Y, predictions_3)
predict_train = model_3.predict(GPU_train_X)
train_error_3 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_3 =  end_time - start_time

start_time=time.process_time_ns()
model_4 = KNeighborsClassifier(n_neighbors= 7, weights= 'distance', algorithm='auto', p=2)
model_4.fit(GPU_train_X, GPU_train_Y)
predictions_4 = model_4.predict(GPU_test_X)
print("Confusion matrix for model_4 is: ")
print(confusion_matrix(GPU_test_Y, predictions_4))
print(classification_report(predictions_4,GPU_test_Y))
print("Accuracy score for the model_4 is: " , accuracy_score(GPU_test_Y, predictions_4))
Error_4 = 1 -  accuracy_score(GPU_test_Y, predictions_4)
predict_train = model_4.predict(GPU_train_X)
train_error_4 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_4 =  end_time - start_time

start_time=time.process_time_ns()
model_5 = KNeighborsClassifier(n_neighbors= 9, weights= 'distance', algorithm='auto', p=2)
model_5.fit(GPU_train_X, GPU_train_Y)
predictions_5 = model_5.predict(GPU_test_X)
print("Confusion matrix for model_5 is: ")
print(confusion_matrix(GPU_test_Y, predictions_5))
print(classification_report(predictions_5,GPU_test_Y))
print("Accuracy score for the model_5 is: " , accuracy_score(GPU_test_Y, predictions_5))
Error_5 = 1 -  accuracy_score(GPU_test_Y, predictions_5)
predict_train = model_5.predict(GPU_train_X)
train_error_5 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_5 =  end_time - start_time

# Learning Curve
test_error = np.array([Error_1, Error_2, Error_3, Error_4, Error_5])
train_error = np.array([train_error_1, train_error_2,train_error_3,train_error_4,train_error_5])
KNN_neighbors = np.array([1,3,5,7,9])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_neighbors, test_error, color = 'blue')
plt.plot(KNN_neighbors,train_error,color='orange')
plt.title("Classification Error with respect to neighbors")
plt.ylabel("Classification Error")
plt.xlabel("KNN_neighbors")
plt.legend(["test error", 'train error'], loc = 'upper right')
plt.figure(figsize = (5,5))

# system time curve exp-1
time = np.array([time_change_1, time_change_2, time_change_3, time_change_4, time_change_5])
KNN_neighbors = np.array(['1-neighbor', '3-neighbor','5-neighbor', '7-neighbor', '9-neighbor'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_neighbors, time, color = 'blue')
plt.title("system time elasped")
plt.ylabel("time_ns")
plt.xlabel("knn neighbors")
plt.legend(["Time elapsed"], loc = 'upper right')
plt.figure(figsize = (5,5))

# KNN Experiements - distance measure
start_time = time.process_time_ns()
model_1_dist = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=1) # Manhattan_distance
model_1_dist.fit(GPU_train_X, GPU_train_Y)
predictions_1_dist = model_1_dist.predict(GPU_test_X)
print("Confusion matrix for model_1_dist is: ")
print(confusion_matrix(GPU_test_Y, predictions_1_dist))
print(classification_report(predictions_1_dist,GPU_test_Y))
print("Accuracy score for the model_1_dist is: " , accuracy_score(GPU_test_Y, predictions_1_dist))
Error_1 = 1 -  accuracy_score(GPU_test_Y, predictions_1_dist)
predict_train = model_1_dist.predict(GPU_train_X)
train_error_1 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_1 =  end_time - start_time

start_time = time.process_time_ns()
model_2_dist = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=2) # Euclidean_distance
model_2_dist.fit(GPU_train_X, GPU_train_Y)
predictions_2_dist = model_2_dist.predict(GPU_test_X)
print("Confusion matrix for model_2_dist is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_dist))
print(classification_report(predictions_2_dist,GPU_test_Y))
print("Accuracy score for the model_2_dist is: " , accuracy_score(GPU_test_Y, predictions_2_dist))
Error_2 = 1 -  accuracy_score(GPU_test_Y, predictions_2_dist)
predict_train = model_2_dist.predict(GPU_train_X)
train_error_2 = 1 - accuracy_score(GPU_train_Y, predict_train)
end_time = time.process_time_ns()
time_change_2 =  end_time - start_time

start_time = time.process_time_ns()
model_3_dist = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=4) # Minkowski_distance
model_3_dist.fit(GPU_train_X, GPU_train_Y)
predictions_3_dist = model_3_dist.predict(GPU_test_X)
print("Confusion matrix for model_3_dist is: ")
print(confusion_matrix(GPU_test_Y, predictions_3_dist))
print(classification_report(predictions_3_dist,GPU_test_Y))
print("Accuracy score for the model_3_dist is: " , accuracy_score(GPU_test_Y, predictions_3_dist))
Error_3 = 1 -  accuracy_score(GPU_test_Y, predictions_3_dist)
end_time = time.process_time_ns()
time_change_3 = end_time - start_time
predict_train = model_3_dist.predict(GPU_train_X)
train_error_3 = 1 - accuracy_score(GPU_train_Y, predict_train)

# Learning Curve with error
test_error = np.array([Error_1, Error_2, Error_3])
train_error = np.array([train_error_1, train_error_2, train_error_3])
KNN_distance_measure = np.array(['Manhattan', 'Euclidean','Minkowski'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_distance_measure, test_error, color = 'blue')
plt.plot(KNN_distance_measure, train_error, color = 'orange')
plt.title("Classification Error with respect to distance measures")
plt.ylabel("Classification Error")
plt.xlabel("KNN_distance_measure")
plt.legend(["test error", "train error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# system time curve
time = np.array([time_change_1, time_change_2, time_change_3])
KNN_distance_measure = np.array(['Manhattan', 'Euclidean','Minkowski'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_distance_measure, time, color = 'blue')
plt.title("system time elasped")
plt.ylabel("Classification Error")
plt.xlabel("Time")
plt.legend(["Time elapsed"], loc = 'upper right')
plt.figure(figsize = (5,5))

# KNN Experiements - weights
start_time = time.process_time_ns()
model_1_weight = KNeighborsClassifier(n_neighbors= 3, weights= 'uniform', algorithm='auto', p=1) # Manhattan_distance
model_1_weight.fit(GPU_train_X, GPU_train_Y)
predictions_1_weight = model_1_weight.predict(GPU_test_X)
print("Confusion matrix for model_1_weight is: ")
print(confusion_matrix(GPU_test_Y, predictions_1_weight))
print(classification_report(predictions_1_weight,GPU_test_Y))
print("Accuracy score for the model_1_weight is: " , accuracy_score(GPU_test_Y, predictions_1_weight))
Error_1 = 1 -  accuracy_score(GPU_test_Y, predictions_1_weight)
end_time = time.process_time_ns()
time_change_1 =  end_time - start_time


start_time = time.process_time_ns()
model_2_weight = KNeighborsClassifier(n_neighbors= 3, weights= 'uniform', algorithm='auto', p=2) # Euclidean_distance
model_2_weight.fit(GPU_train_X, GPU_train_Y)
predictions_2_weight = model_2_weight.predict(GPU_test_X)
print("Confusion matrix for model_2_weight is: ")
print(confusion_matrix(GPU_test_Y, predictions_2_weight))
print(classification_report(predictions_2_weight,GPU_test_Y))
print("Accuracy score for the model_2_weight is: " , accuracy_score(GPU_test_Y, predictions_2_weight))
Error_2 = 1 -  accuracy_score(GPU_test_Y, predictions_2_weight)
end_time = time.process_time_ns()
time_change_2 =  end_time - start_time

start_time = time.process_time_ns()
model_3_weight = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=1) # Manhattan_distance
model_3_weight.fit(GPU_train_X, GPU_train_Y)
predictions_3_weight = model_3_weight.predict(GPU_test_X)
print("Confusion matrix for model_3_weight is: ")
print(confusion_matrix(GPU_test_Y, predictions_3_weight))
print(classification_report(predictions_3_weight,GPU_test_Y))
print("Accuracy score for the model_3_dist is: " , accuracy_score(GPU_test_Y, predictions_3_weight))
Error_3 = 1 -  accuracy_score(GPU_test_Y, predictions_3_weight)
end_time = time.process_time_ns()
time_change_3 = end_time - start_time

start_time = time.process_time_ns()
model_4_weight = KNeighborsClassifier(n_neighbors= 3, weights= 'distance', algorithm='auto', p=2) # euclidean_distance
model_4_weight.fit(GPU_train_X, GPU_train_Y)
predictions_4_weight = model_4_weight.predict(GPU_test_X)
print("Confusion matrix for model_4_weight is: ")
print(confusion_matrix(GPU_test_Y, predictions_4_weight))
print(classification_report(predictions_4_weight,GPU_test_Y))
print("Accuracy score for the model_4_dist is: " , accuracy_score(GPU_test_Y, predictions_4_weight))
Error_4 = 1 -  accuracy_score(GPU_test_Y, predictions_4_weight)
end_time = time.process_time_ns()
time_change_4 = end_time - start_time

# Learning Curve with error
test_error = np.array([Error_1, Error_2, Error_3, Error_4])
KNN_weights_measure = np.array(['Manhattan_Uniform', 'Euclidean_Uniform','Manhattan_distance', 'Euclidean_dstance'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_weights_measure, test_error, color = 'blue')
plt.title("Classification Error with respect to weights")
plt.ylabel("Classification Error")
plt.xlabel("KNN_weights_measure")
plt.legend(["test error"], loc = 'upper right')
plt.figure(figsize = (5,5))

# system time curve
time = np.array([time_change_1, time_change_2, time_change_3, time_change_4])
KNN_weights_measure = np.array(['Manhattan_Uniform', 'Euclidean_Uniform','Manhattan_distance', 'Euclidean_dstance'])
plt.style.use('seaborn-whitegrid')
plt.plot(KNN_weights_measure, time, color = 'blue')
plt.title("system time elasped")
plt.ylabel("time_ns")
plt.xlabel("Time")
plt.legend(["Time elapsed"], loc = 'upper right')
plt.figure(figsize = (5,5))
