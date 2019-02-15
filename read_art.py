#!/usr/bin/env python3
# Simple program to read netcdf data

import sys, argparse
import numpy as np
import pyart

def usage():
    print('reader.py [-h] -i <inputfile>')
    sys.exit(0)
            
def main(argv):
    parser = argparse.ArgumentParser(description = "Read a given netcdf file")
    parser.add_argument('-i', action="store", dest = "input_file")

    results = parser.parse_args(argv)
    if not results.input_file:
        print('The -i optionis required')
        usage()

    # radar = pyart.io.read_cfradial(results.input_file)
    radar = pyart.io.read(results.input_file)
    print(radar.altitude['data'].shape)
    
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape = (1, 300, 300),
        grid_origin_alt = radar.altitude['data'][0],
        grid_origin = (radar.latitude['data'][0], radar.longitude['data'][0]),  

        grid_limits = ((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
        fields = ['ZZ'])
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
        

            


    
