#!/usr/bin/env python3

# Bruno Melli 2/11/19
#
# h5 to dataset
# - Grab and combine a given percentage of X and Y data from
# - the given h5 files (not normalized)
#
# Problem:  The data is unbalanced. Many more false than true
#           To work around the problem, just use as many falses as
#              there are trues
#     
# - Randommize
# - Write out 2 files:
#    X and Y with that percentage of data
#    X and Y with the data not used 
#      (so that it can be further used for testing later)

import numpy as np
import sys, argparse
import h5py
from utils import *

def usage():
    print('h52ds.py [-h] -i <dir|inputfile> [-o <output_prefix] [-p <percentage> ]')
    sys.exit(0)

def main(argv):
    parser = argparse.ArgumentParser(description = "Grab a ratio of inputs from the given files, randomize, write out h5")
    parser.add_argument('-i', '--input', action="store", dest = "input_file")
    parser.add_argument('-o', '--output', action="store",
                        dest = "output_prefix",
                        default = 'dataset.h5')
    parser.add_argument('-p', '--percentage', action="store",
                        dest = "percentage", type = float,
                        default = 50)

    args = parser.parse_args(argv)
    if not args.input_file:
        print('The -i option is required')
        usage()

    combined_X = np.empty( (0, 0) )
    combined_Y = np.empty( (0,) )

    testing_X = np.empty( (0, 0) )
    testing_Y = np.empty( (0,) )

    file_list = expand_path(args.input_file, '.*\.h5')
    for path in file_list:
        print ('Reading ', path)
        dataset = h5py.File(path, 'r')
        
        X_orig = np.array(dataset['X'][:])
        Y_orig = np.array(dataset['Y'][:])
        dataset.close()

        print('Original X: ', X_orig.shape, ', Y: ', Y_orig.shape)

        ones_idx = np.argwhere(Y_orig == 1)
        zeros_idx = np.argwhere(Y_orig == 0)

        # How many positive do we have
        one_count = ones_idx.size
        print('one_count: ', one_count)
        print('zero_count: ', zeros_idx.size)
        
        # Keep all the positive entries
        X = X_orig[:, ones_idx].squeeze()
        Y = np.ones(one_count, dtype = int).squeeze()

        print('---  Ones X: ', X.shape, ', Y: ', Y.shape)        
        
        # Keep only one_count number of negative entries
        zero_X = X_orig[:, zeros_idx].squeeze()
        print('--- Zeros X: ', zero_X.shape)
        p = np.random.permutation(zero_X.shape[1])
        zero_subset = zero_X[:, p]
        zero_subset = zero_subset[:, 0:one_count]
        print('--- zero_subset: ', zero_subset.shape)
        
        X = np.append(X, zero_subset, axis = 1)
        Y = np.append(Y, np.zeros(one_count))

        # Randomize so not all the 0s and all the 1s are next to each others
        

        print('--- Combined X: ', X.shape, ', Y: ', Y.shape)        
        
        p = np.random.permutation(X.shape[1])
        X = X[:, p]
        Y = Y[p]

        # Shape of X is (# attrs, # obs)
        # Shape of Y is (# obs, )

        num = int(X.shape[1] * args.percentage / 100)

        if combined_X.shape[0]:
            combined_X = np.append(combined_X, X[:, 0:num], axis = 1)
            combined_Y = np.append(combined_Y, Y[0:num])
            testing_X = np.append(testing_X, X[:, num:], axis = 1)
            testing_Y = np.append(testing_Y, Y[num:])
        else:
            combined_X = X[:, 0:num]
            combined_Y = Y[0:num]
            testing_X = X[:, num:]
            testing_Y = Y[num:]
            
    # Get a random permutation  TODO revisit this as this might need to be done in the loop
    p = np.random.permutation(combined_X.shape[1])
    random_x = combined_X[:, p]
    random_y = combined_Y[p]

    print ('randomX: ', random_x.shape)
    print ('randomY: ', random_y.shape)
    print ('testingX: ', testing_X.shape)
    print ('testingY: ', testing_Y.shape)
    
    with h5py.File(args.output_prefix + 'train.h5', 'w') as f:
        dset = f.create_dataset('X', random_x.shape, data = random_x)
        dset = f.create_dataset('Y', random_y.shape, data = random_y)
    with h5py.File(args.output_prefix + 'test.h5', 'w') as f:
        dset = f.create_dataset('X', testing_X.shape, data = testing_X)
        dset = f.create_dataset('Y', testing_Y.shape, data = testing_Y)
        
if __name__ == '__main__':
    main(sys.argv[1:])
