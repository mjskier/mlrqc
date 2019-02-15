#!/usr/bin/env python3
import numpy as np

arr1 = np.arange(18).reshape((3, 6))
arr2 = arr1 * 3
print (arr1)
print ('---')
print (arr2)

arr3 = np.append(arr1, arr2, axis = 1)

print ('---')
print (arr3)

# Numpy arrays don't grow automatically

# a = np.arange(5)
# a[7] = 50
# print(a)

# Try shuffle and permutations

# arr = np.arange(18).reshape((3, 6))
# print (arr)
# print ('---')
# print (np.random.permutation(arr.T).T)
# print ('---')
# arr = arr.T
# np.random.shuffle(arr)
# arr = arr.T
# print (arr)

# Try splitting into train and test

# x = numpy.random.rand(100, 5)
# numpy.random.shuffle(x)
# training, test = x[:split,:], x[split:,:]

# print("train: ", training.shape, ", test: ", test.shape)
