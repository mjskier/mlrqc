#!/usr/bin/env python3

# Simple program to preprocess radar sweep files
#
# - read netcdf radar sweep file(s)
# - remove obs were VV is fillVal
# - Add altitude field
# - Compute Y (VV == VG for that ob)
# - Randomize
# - Output X and Y in an h5 file

import numpy as np
import sys, argparse
import h5py

from utils import *

def usage():
    print('reader.py [-h] -i <dir|inputfile> [-o <output_file]')
    sys.exit(0)

def main(argv):
    parser = argparse.ArgumentParser(description = "Read a given set of netcdf file, convert to one .hd5 file")
    parser.add_argument('-i', action="store", dest = "input_file")
    parser.add_argument('-o', action="store", dest = "output_file")

    args = parser.parse_args(argv)
    if not args.input_file:
        print('The -i option is required')
        usage()
        
    if not args.output_file:
        args.output_file = '/tmp/data.h5'

    X, Y = load_netcdf(args.input_file)
    
    with h5py.File(args.output_file, 'w') as f:
        dset = f.create_dataset("X", X.shape, data = X)
        dset = f.create_dataset("Y", Y.shape, data = Y)
    
if __name__ == '__main__':
    main(sys.argv[1:])
