#!/usr/bin/env python3

# Simple program to preprocess radar sweep files
#
# - read netcdf radar sweep file(s)
# - output X and Y in an h5 file

import numpy as np
import sys, argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn import svm

from utils import *

def usage():
    print('radar_qc .py [-h] [-l <prefix>] -i <dir|inputfile> [-s <split>] [-p <prefix>]')
    sys.exit(0)

class Model:

    def load_model(prefix):
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

    def save_model(prefix):

        # Serialize model to YAML
    
        model_yaml = self.model.to_yaml()
        with open(prefix + '.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

            # Serialize weights to HDF5
    
            self.model.save_weights(prefix + '.h5')
    
    def create_model(input_size):
        print('Base create_model needs to be specialized')
        return False

    def train(X_train, Y_train, prefix):
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
    
    def evaluate(X_test, Y_test):
        print('Evaluating using a ', self.identify())
        return self.model.evaluate(X_test, Y_test, batch_size = 128)

class NeuralNetwork(Model):
    def __init__(self, idim):
        print('Creating Neural Network model')
        self.create_model(idim)

    def create_model(self, idim):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim = idim, activation = 'relu'))
        self.model.add(Dropout(0.2))     # Avoid over fitting
        self.model.add(Dense(8, activation = 'relu'))
        self.model.add(Dense(1, activation = 'sigmoid'))

        # self.model.compile(loss = 'binary_crossentropy',
        # optimizer = 'rmsprop',
        self.model.compile(loss = 'mean_squared_error',
                           optimizer = 'adam',
                           metrics = ['accuracy'])
        
    def identify(self):
         return 'Neural Network'
        
class LogisticRegression(Model):
    
    def __init__(self, idim):
        print('Creating Logistic Regression model. Input dim: ', idim)
        self.create_model(idim)

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
        self.create_model(idim)

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
        self.create_model()

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

    def predict(self, X_test, Y_test):
        print('Predicting using a Support Vactor Machine')
        self.model.score(X_test, Y_test)

# Model object factory

def create_model(kind, dim):
    model = {
        'nn':   lambda d: NeuralNetwork(d),
        'lr':   lambda d: LogisticRegression(d),
        'l1l2': lambda d: LogisticRegressionL1L2(d),
        'svm':  lambda d: SupportVectorMachine(d)
    }[kind](dim)
    return model

def main(argv):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description = 'Read a given netcdf file')
    parser.add_argument('-i', '--input', action = 'store',
                        help = 'Input file (dataset)',
                        dest = 'input_file', required = True)
    parser.add_argument('-s', '--split', action = 'store', dest = 'split',
                        help = 'Percentare to split dataset into train and test',
                        default = 90, type = int)
    parser.add_argument('-l', '--load', action = 'store', dest = 'load',
                        help = 'Load an existing model from this prefix')
    parser.add_argument('-m', '--model', action = 'store', dest = 'model',
                        choices = { 'nn', 'lr', 'l1l2', 'svm' },
                        help = 'nn: Neural network, ' +
                        'svm: Support Vector Machine, ' +
                        'lr: Logistic Regression, ' +
                        'l1l2: Logistic regression with L1 and L2 regularization',
                        default = 'nn')
    parser.add_argument('-p', '--prefix', action = 'store', dest = 'prefix',
                        default = 'my_model')
    
    args = parser.parse_args(argv)
    print(args)
    
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

    if args.load:
        model = load_model(args.load)
    else:
        model = create_model(args.model, X_train.shape[1])

    if model == None:
        print('Error creating a ', args.model, ' model. Exiting')
        sys.exit(1)

    # Train the model
    
    model.train(X_train, Y_train, args.prefix)

    # evaluate the model
    
    print('Evaluating the model...')
    scores = model.evaluate(X_test, Y_test, batch_size = 128)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

if __name__ == '__main__':
    main(sys.argv[1:])
    


    
