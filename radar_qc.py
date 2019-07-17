#!/usr/bin/env python3

# Bruno Melli 2/11/19
#
# Simple program to preprocess radar sweep files
#
# - read netcdf radar sweep file(s)
# - output X and Y in an h5 file

import numpy as np
import sys, argparse
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import model_from_yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from utils import *

def usage():
    print('radar_qc .py [-h] [-l <prefix>] -i <dir|inputfile> [-s <split>] [-p <prefix>]')
    print('                  [-m <model>] [-d <model def>]')
    sys.exit(0)

class Model:

    def load_model(self, prefix):
        print('Loading model from saved ', prefix + '.yaml')
        
        yaml_file = open(prefix + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)

        # Load saved weights into the model

        loaded_model.load_weights(prefix + '.h5')
    
        # Compile it TODO args should match the ones used in create_and_train
        loaded_model.compile(loss = 'binary_crossentropy',
                             optimizer = 'rmsprop',
                             metrics = ['accuracy'])

        self.model = loaded_model

    def save_model(self, prefix):

        # Serialize model to YAML
    
        model_yaml = self.model.to_yaml()
        with open(prefix + '.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

            # Serialize weights to HDF5
    
            self.model.save_weights(prefix + '.h5')
    
    def create_model(self, input_size):
        print('Base create_model needs to be specialized')
        return False

    def train(self, X_train, Y_train, prefix):
        print('Training a ', self.identify())
        
        # Add some callbacks
    
        early_stop = EarlyStopping(monitor = 'val_loss',
                                   min_delta = 0.0001,
                                   patience  = 20,
                                   # 2.2.2 restore_best_weights = True,
                                   verbose = 1)

        checkpoint = ModelCheckpoint(filepath = prefix + '_best_model.h5',
                                     monitor = 'val_loss',
                                     save_best_only = True)

        tensor_board = TensorBoard(histogram_freq = 2, batch_size = 32,
                                   write_graph = True,
                                   write_grads = True,
                                   write_images = True)

        callback_list = [early_stop, checkpoint, tensor_board]
    
        history = self.model.fit(X_train, Y_train, validation_split = 0.33,
                                 callbacks = callback_list,
                                 epochs = 64, batch_size = 32,
                                 verbose = 0)

        # Write some data to be visualized later
    
        with open(prefix + '_history.txt', 'w') as fp:
            print(history.history, file = fp)
    
    def evaluate(self, X_test, Y_test):
        print('Evaluating using a ', self.identify())
        return self.model.evaluate(X_test, Y_test)

    def predict_classes(self, X_test):
        return self.model.predict_classes(X_test)

    def show_prediction_metrics(self, Y_test, predictions):
        print('Accuracy: ', metrics.accuracy_score(Y_test, predictions))

        matrix = confusion_matrix(Y_test, predictions)
        print('Matrix type: ', type(matrix), ' shape: ', matrix.shape)
        print('Un-normalized Confusion Matrix')
        print(matrix)
        
        print('Normalized Confusion Matrix')
        matrix = matrix.astype('float') / matrix.sum(axis = 1)[:, np.newaxis]
        print(matrix)                           
    
        print(pd.crosstab(Y_test, predictions,
                          rownames=['True'], colnames=['Predicted'], margins=True))

        print(classification_report(Y_test, predictions))
    
class NeuralNetwork(Model):
    
    def __init__(self, idim):
        print('Creating Neural Network model')

    def create_model(self, idim):
        # Default hard coded model
        self.model = Sequential()
        #self.model.add(Dense(18, input_dim = idim, activation = 'relu'))
        self.model.add(Dense(18, input_dim = idim))
        self.model.add(LeakyReLU(alpha = 0.1))
        self.model.add(Dropout(0.2))     # Avoid over fitting
        #self.model.add(BatchNormalization())
        # self.model.add(Dense(8, activation = 'relu'))
        self.model.add(Dense(8))
        self.model.add(LeakyReLU(alpha = 0.1))
        self.model.add(Dropout(0.2))     # Avoid over fitting
        #self.model.add(BatchNormalization())        
        self.model.add(Dense(1, activation = 'sigmoid'))
        # self.model.compile(loss = 'binary_crossentropy',
        # optimizer = 'rmsprop',
        
        self.model.compile(loss = 'mean_squared_error',
                           optimizer = 'adam',
                           metrics = ['accuracy'])
        
    def identify(self):
         return 'Neural Network'

class RandomForest(Model):

    def __init__(self, idim):
        print('Creating Random Forest model. Input dim: ', idim)

    def create_model(self, idim): # TODO parameterise n_estimators
        self.model = RandomForestClassifier(n_estimators = 100, n_jobs = -1,
                                            random_state = 50, oob_score = True,
                                            min_samples_leaf = 50)


        return 'Random Forest'
    
    def save_model(self, prefix):
        with open(prefix + '.pickle', 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file, pickle.HIGHEST_PROTOCOL)

    def load_model(self, prefix):
        print('Loading model from saved ', prefix + '.pickle')
        with open(prefix + '.pickle', 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)
            
    def train(self, X_train, Y_train, prefix):
        print('Training Random Forest model')
        self.model.fit(X_train, Y_train)
        
        feature_imp = pd.Series(self.model.feature_importances_,
                                index=field_names()).sort_values(ascending=False)
        print('Feature importance: ', feature_imp)
        
    def predict_classes(self, X_test):
        return self.model.predict(X_test)

class LogisticRegression(Model):

    def __init__(self, idim):
        print('Creating Logistic Regression model. Input dim: ', idim)

    def create_model(self, idim):
        self.model = Sequential()
        self.model.add(Dense(1, input_dim = idim, activation = 'sigmoid'))
        self.model.compile(loss = 'binary_crossentropy',
                           optimizer = 'rmsprop',
                           metrics = ['accuracy'])
        
    def identify(self):
        return 'Logistic Regression'
        
class LogisticRegressionL1L2(Model):
    def __init__(self, idim):
        print('Creating Logistic Regression with L1 and L2 regularization model')

    def create_model(self, idim):
        from keras.regularizers import l1_l2
        reg = l1_l2(l1 = 0.01, l2 = 0.1)
    
        self.model = Sequential()
        self.model.add(Dense(1, input_dim = idim, activation = 'sigmoid',
                             kernel_regularizer = reg))
        self.model.compile(loss = 'binary_crossentropy',
                           optimizer = 'rmsprop',
                           metrics = ['accuracy'])
        
    def identify(self):
        return 'Logistic Regression with L1 L2 regularisation'
        
class SupportVectorMachine(Model):
    def __init__(self, idim):
        print('Creating a Support Vector Machine')

    def create_model(self):
        self.model = svm.SVC(kernel = 'linear', C = 0.01, verbose = True)

    def identify(self):
        return 'Support Vector Machine'

    def load_model(self, prefix):
        print('SVM::load_model not implemented yet')
        return None

    def save_model(self, prefix):
        print('SVM::save_model not implemented yet')
        return None

    def train(self, X_train, Y_train, prefix):
        print('Training a Support Vector Machine model')
        self.model.fit(X_train, Y_train)

    def predict_classes(self, X_test, Y_test):
        print('Predicting using a Support Vactor Machine')
        self.model.score(X_test, Y_test)

# Model object factory

def new_model(kind):
    model = {
        'nn':   lambda d: NeuralNetwork(d),
        'lr':   lambda d: LogisticRegression(d),
        'l1l2': lambda d: LogisticRegressionL1L2(d),
        'rf':   lambda d: RandomForest(d),
        'svm':  lambda d: SupportVectorMachine(d)
    }[kind](0)
    return model

def train_model(args):
                    
    # Load the dataset, split into 90% train and 10% test
    
    X, Y = load_h5(args.input_file)
    X, min_max = normalize(X)

    # TODO: Can Keras only run on feature rows?
    
    X = X.T
    Y = Y.T
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10)
    
    # Print some info
    
    print('X_train: ', X_train.shape)
    trues = np.count_nonzero(Y)
    print('True:  ', trues)
    print('False: ', Y.shape[0] - trues) 

    # Create model
    
    model = new_model(args.model)
    print('Model: ', type(model))
    
    if args.model_to_load: # load with trained weight
        model.load_model(args.model_to_load)
    else: 
        model.create_model(X_train.shape[1])
        model.train(X_train, Y_train, args.prefix)
        model.save_model(args.prefix)

    if model == None:
        print('Error creating a ', args.model, ' model. Exiting')
        sys.exit(1)

    # evaluate the model
    
    predictions = model.predict_classes(X_test).reshape(-1)
    model.show_prediction_metrics(Y_test, predictions)

    # # debug: Try to run on the testing set to see if we are way off like
    # #        after loading the model in run_model

    # tX, tY = load_h5((args.input_file).replace('train', 'test'))
    # tX, dont_care = normalize(tX)
    # tX = tX.T
    # tY = tY.T

    # t_predictions = model.predict_classes(tX).reshape(-1)
    # model.show_prediction_metrics(tY, t_predictions)
    
def run_model(args):
    # Load the dataset

    X, Y = load_h5(args.input_file)
    X, min_max = normalize(X)
    
    X = X.T
    Y = Y.T

    # Load model
    
    model = new_model(args.model) # TODO deduct the model type from the saved model
    print('Model: ', type(model))
    model.load_model(args.model_to_load)

    predictions = model.predict_classes(X).reshape(-1)
    model.show_prediction_metrics(Y, predictions)
    
    # TODO Write the results
    # Current thinking: Reading from and writing to the original radar file is done from an external script
    print('Need to implement the writing out of the data')
    
def main(argv):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description = 'Read a given netcdf file')
    parser.add_argument('-a', '--action', action = 'store',
                        help = 'One of: train, eval',
                        dest = 'action', default = 'train')
    parser.add_argument('-i', '--input', action = 'store',
                        help = 'Input file (dataset)',
                        dest = 'input_file', required = True)
    parser.add_argument('-s', '--split', action = 'store', dest = 'split',
                        help = 'Percentare to split dataset into train and test',
                        default = 90, type = int)
    parser.add_argument('-l', '--load', action = 'store', dest = 'model_to_load',
                        help = 'Load an existing model from this prefix')
    parser.add_argument('-m', '--model', action = 'store', dest = 'model',
                        choices = { 'nn', 'lr', 'l1l2', 'rf', 'svm' },
                        help = 'nn: Neural network, ' +
                        'lr: Logistic Regression, ' +
                        'rf: Random Forrest, ' +
                        'l1l2: Logistic regression with L1 and L2 regularization',
                        default = 'nn')
    parser.add_argument('-p', '--prefix', action = 'store', dest = 'prefix',
                        help = 'Use this prefix to save model and weights',
                        default = 'my_model')
    
    args = parser.parse_args(argv)
    print(args)

    if args.action == 'eval':
        run_model(args)
    elif args.action == 'train':
        train_model(args)
    else:
        print('Unknown action ', args.action)
    
if __name__ == '__main__':
    main(sys.argv[1:])
