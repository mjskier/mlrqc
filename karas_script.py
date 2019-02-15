import numpy as np

from keras.models import Model Sequential
from data_generator import DataGenerator
from keras.layers import Input, Dense

# Parameters
params  =  {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition  =  # IDs
labels  =  # Labels

# Generators

training_generator  =  DataGenerator(partition['train'], labels, **params)
validation_generator  =  DataGenerator(partition['validation'], labels, **params)

num_features  =  5 # DBZ, ...
num_units_1  =  5
num_units_2  =  4
num_outputs  =  1

# Input tensor

inp_tensor  =  Input(shape = (num_features,))

# Design model (functional API

hidden1_out  =  Dense(units = num_units_1)(inp_tensor)
hidden2_out  =  Dense(units = num_units_2)(hidden1_out)
final_out  =  Dense(units = num_outputs)(hidden2_out)

model  =  Model(inputs = inp_tensor, outputs = final_out)

model.compile(metrics = ['accuracy'])

# Train model on data generated batch by batch by a Python generator
# or an instance of Sequence

model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    use_multiprocessing = True,
                    workers = 6)

# Check how we did with evaluate()
# Test data with predict(x, batch_size=None, verbose=0, steps=None)
#
# Check difference between predict and test

