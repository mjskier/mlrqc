#!/usr/bin/env python3

# Bruno Melli 2/11/19
#
# h5 to dataset
# - Grab and combine a given percentage of X and Y data from
# - the given normalized h5 files
# - Randommize
# - Write out new h5 file

import numpy as np
import sys, argparse
import h5py
from utils import *

def usage():
    print('h52ds.py [-h] -i <dir|inputfile> [-o <output_file]')
    sys.exit(0)

def main(argv):
    parser = argparse.ArgumentParser(description = "Grab a ratio of inputs from the given files, randomize, write out h5")
    parser.add_argument('-i', '--input', action="store", dest = "input_file")
    parser.add_argument('-o', '--output', action="store",
                        dest = "output_file",
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
    
    file_list = expand_path(args.input_file, '.*\.h5')
    for path in file_list:
        print ('Reading ', path)
        dataset = h5py.File(path, 'r')
        
        X = np.array(dataset['X'][:])
        Y = np.array(dataset['Y'][:])
        dataset.close()
        
        # Shape of X is (# attrs, # obs)
        # Shape of Y is (# obs, )

        num = int(X.shape[1] * args.percentage / 100)

        if combined_X.shape[0]:
            combined_X = np.append(combined_X, X[:, 0:num], axis = 1)
            combined_Y = np.append(combined_Y, Y[0:num])
        else:
            combined_X = X[:, 0:num]
            combined_Y = Y[0:num]
            
    # Get a random permutation
    p = np.random.permutation(combined_X.shape[1])
    random_x = combined_X[:, p]
    random_y = combined_Y[p]

    print ('randomX: ', random_x.shape)
    print ('randomY: ', random_y.shape)
    
    with h5py.File(args.output_file, 'w') as f:
        dset = f.create_dataset('X', random_x.shape, data = random_x)
        dset = f.create_dataset('Y', random_y.shape, data = random_y)
        

if __name__ == '__main__':
    main(sys.argv[1:])
    

        
                            

                                     

