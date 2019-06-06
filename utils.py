import math
import os, sys
import numpy as np
from scipy import ndimage
import netCDF4 as nc4
import re
import h5py

# Global constant so that we don't recompute that every time
# we call ...

EarthRadiusKm = 6375.636
EarthRadiusSquare = 6375.636 * 6375.636
DegToRad = 0.01745329251994372

# For doing average of neighbor values

avg_mask = np.ones((3, 3))
avg_mask[1, 1] = 0

def computeHtKmAirborn(elDeg, slantRangeKm, aircraftHtKm):
    elRad = elDeg * DegToRad
    term1 = slantRangeKm * slantRangeKm + EarthRadiusSquare
    term2 = slantRangeKm * EarthRadiusKm * math.sin(elRad)
    return aircraftHtKm - EarthRadiusKm + math.sqrt(term1 + term2)

def normalize(X):
    # Normalize values

    print('Normalizing value')
    min_maxs = np.empty((X.shape[0], 2))
    
    # print('X: ', X.shape)
    # print('min_maxs shape: ', min_maxs.shape)

    # Remember min and max we used so that we can normalize test data later
    
    for field in range(X.shape[0]):
        # print('X[', field, ']: ', X[field].shape)
        min = X[field].min()
        max = X[field].max()
        denominator = max - min
        # print('min_max[0]: ', min_maxs[0].shape)
        
        min_maxs[0] = X[field].min()
        min_maxs[1] = X[field].max()
        X[field] = (X[field] - min) / denominator

    return X, min_maxs

    
# loc is a file? load just this file
# loc is a dir? load all files in that dit

def expand_path(path, pattern = None):
    
    if os.path.isfile(path):
        return [ path ]
    
    file_list = []
    pattern = pattern or ".*"
    patt = re.compile(pattern)
    
    for root, dirs, files in os.walk(path, topdown = False):
        for name in files:
            if patt.match(name):
                file_list.append(os.path.join(root, name))        
    return file_list

# Just as an image would be X * Y * 3 (3 values for RGB)
# our "images are X * Y * (size_of(my_vars) + 1)
#    +1 to add the altitude
#
# 

# This is super slow as it has to open and close all the files.
# But it gives me the exact size I need to create an numpy array
# that doesn't have to be reallocated each time file data is appended
# to it...
# OK tradoff since we run this only once to convert to hdf5...

def get_dim(file_list):
    max_time = 0
    max_range = 0
    
    for path in file_list:
        nc_ds = nc4.Dataset(path, 'r')
        max_time = max(max_time, nc_ds.dimensions['time'].size)
        max_range = max(max_range, nc_ds.dimensions['range'].size)
        nc_ds.close()
    return max_time, max_range


# Currently not used

# Load a dataset so that each file is a slice of 6 variables.
# Each variable will be a 2d plane

def not_used_load_dataset_4d(loc, pattern = None):

    my_vars = [ 'ZZ', 'VV', 'SW', 'NCP', 'VG' ]
    result_var = 'VG'

    file_list = expand_path(loc, pattern)
    num_files = len(file_list)

    dim = get_dim(file_list)    # (max x, max y)
    print ('--- dim: ', dim)
    dim = (num_files,) + (6,) + dim
    
    X = np.empty(dim)
    print("X: ", X.shape)

    f_count = 0
    for path in file_list:
        print ('Reading ', path)
        nc_ds = nc4.Dataset(path, 'r')
        v_count = 0
        for var in my_vars:
            my_var = nc_ds.variables[var]
            X[f_count,v_count,:] = my_var[:]
        nc_ds.close()

    return X


#
# Slit X and Y into X_train, Y_train, X_test, Y_test
#

def split_dataset(X, Y, split = 90):
    count = Y.shape[0]
    cutoff = int(Y.shape[0] * split / 100)

    X_train, X_test = X[:, :cutoff], X[:, cutoff:]
    Y_train, Y_test = Y[:cutoff], Y[cutoff:]
    
    return X_train, X_test, Y_train, Y_test


# Fill expected result by comparing VV to VG
# O: original field
# E: Edited field
# return: numpy array of booleans: True of the values are the same

def fill_expected(O, E):
    return (O == E).astype(int)

# Compute neighbor average for a given variable
# var: 2D variable from the netcdf file
# max_x, max_y: 2D dimensions
# return: numpy array of averages

def compute_avgs(var, max_x, max_y, fill_val):
    return ndimage.generic_filter(var, np.nanmean, size = 3, footprint = avg_mask,
                                  mode = 'constant', cval = np.NaN)

# Simple for now
# X: (9, *) array  (ZZ, VV, SW, NCP, ALT, AZZ, AVV, ASW, ANCP)
# Y: (1, *) array
#
# TODO pass fields as arguments...

def field_names():
    return ['ZZ', 'VV', 'SW', 'NCP', 'ALT', 'AZZ', 'AVV', 'ASW', 'ANCP']

def load_netcdf(loc, pattern = None):
    
    my_vars = [ 'ZZ', 'VV', 'SW', 'NCP', 'VG']
    my_avgs = [ 'AZZ', 'AVV', 'ASW', 'ANCP' ]
    result_var = 'VG'
    alt_index = len(my_vars)        # Index of altitude
    avg_offset = alt_index + 1      # Averages will be after the altitude
    
    file_list = expand_path(loc, pattern)
    num_files = len(file_list)

    max_x, max_y = get_dim(file_list)
    flat_size = max_x * max_y
    num_cols = num_files * flat_size

    dim = (len(my_vars) + 1 + len(my_avgs), num_cols)
    
    X = np.empty(dim)
    VG = np.empty(num_cols)
    
    ob_index = 0
    fill_val = None
    
    for path in file_list:
        print ('Reading ', path)
        nc_ds = nc4.Dataset(path, 'r')

        # read the variables we need to compute altitude
        max_time = nc_ds.dimensions['time'].size
        max_range = nc_ds.dimensions['range'].size
        
        plane_alts = nc_ds.variables['altitude'] # (time)
        ray_angles = nc_ds.variables['elevation'] # (time)
        ranges = nc_ds.variables['range'] # (time)
        
        # Read the "feature" variables
        
        for var in range(len(my_vars)):
            my_var = nc_ds.variables[my_vars[var]]
            fill_val = getattr(my_var, '_FillValue')

            # Flatten it and append to the X array
            one_d = (my_var[:]).reshape(-1)
            X[var, range(ob_index, ob_index + one_d.size)] = one_d

            # Compute the neighbor average, and append it to X
            if var == 4:        # Don't average VG
                continue
            print('Computing averages for ', my_vars[var])
            var_avrg = compute_avgs(my_var, max_time, max_range, fill_val)
            one_d = (var_avrg[:]).reshape(-1)
            X[var + avg_offset, range(ob_index, ob_index + one_d.size)] = one_d
            
        # Compute and add the height

        print('Computing altitudes for ', max_time * max_range, ' entries')
        heights = np.empty( (max_time, max_range) )
        
        for time_idx in range(max_time):
            for range_idx in range(max_range):
                tangle = float(ray_angles[time_idx].data)
                trange = float(ranges[range_idx].data)
                talt   = float(plane_alts[time_idx].data)
                
                heights[time_idx, range_idx] = computeHtKmAirborn(
                    tangle, trange, talt)

        one_d = (heights[:]).reshape(-1)
        X[alt_index, range(ob_index, ob_index + one_d.size)] = one_d
        nc_ds.close()
        ob_index += one_d.size

    # Remove columns where VV is fill_val (Should that be done earlier?)
    #  as we are processing each file?
    
    X = X[:, X[0] != fill_val]

    # Randomize the data

    print ('Randomizing the data')
    X = np.random.permutation(X.T).T
    
    # print("X after: ", X.shape)
    
    Y = fill_expected(X[my_vars.index('VV')],
                      X[my_vars.index('VG')])

    # Delete the VG row from X (since it is now Y)
    X = np.delete(X, my_vars.index('VG'), 0)

    return X, Y

#
# Load X and Y from the given h5 file
#

def load_h5(loc):
    dataset = h5py.File(loc, 'r')
    X = np.array(dataset['X'][:])
    Y = np.array(dataset['Y'][:])
    
    print('X: ', X.shape)
    print('Y: ', Y.shape)

    return X, Y

#
# Precision, recall, and f1_score
# Not implemented yet
#

def precision(matrix):
    return 1

def recall(matrix):
    return 1

def f1_score(prec, recl):
    return 2 * prec * recl / (prec + recl)




