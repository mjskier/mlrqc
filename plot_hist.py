#!/usr/bin/env python3

# - Extract and plot the history from a Keras run
# - This assumes that the model metrics were printed
# - Bruno Melli 2/12/19

import matplotlib.pyplot as plt
import sys, argparse
import re

def plus (a_list, a_num):
    return [x + a_num for x in a_list]

def plotit(history, dest, plot):
    
    # Plot training & validation accuracy values
    
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if plot:
        plt.show()
    plt.savefig(dest + '_accuracy.png')
    
    # Plot training & validation loss values
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if plot:
        plt.show()
    plt.savefig(dest + '_loss.png')

def read_history(path):
    patt = re.compile('^{')
    
    with open(path) as fp:
        for line in fp:
            if patt.match(line):
                the_dict = eval(line)
                print('dict: ', the_dict)
                return the_dict
    return None

def main(argv):
    parser = argparse.ArgumentParser(description = 'Plot training and validation from a file')
    parser.add_argument('-i', '--input', action = 'store',
                        dest = 'input_file', required = True)
    parser.add_argument('-p', '--prefix', action = 'store', dest = 'prefix',
                        default = 'my_graphs')
    parser.add_argument('-s', '--show', action = 'store', dest = 'show', default= True)

    args = parser.parse_args(argv)
    history = read_history(args.input_file)
    if history != None:
        print('history: ', type(history))
        plotit(history, args.prefix, args.show)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
