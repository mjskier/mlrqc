#!/usr/bin/env python3

import numpy as np
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow


def draw_feed_forward(ax, num_node_list):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): number of nodes in each layer.
    '''

    my_vars = [ 'ZZ', 'VV', 'SW', 'NCP', 'ALT' ]
    
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]
    y_list = 1.5 * np.arange(len(num_node_list))

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        nodes = node_sequence(b, n, center=(0, y))
        
        if kind == 'nn.input':
            for index in range(n):
                nodes[index].text(my_vars[index], 'center', fontsize = 12)
        elif kind == 'nn.output':
            nodes[0].text('VV == VG', 'center', fontsize = 10)
            
        seq_list.append(nodes)

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)


def real_bp():
    with DynamicShow((6, 6), '_feed_forward.png') as d:
        draw_feed_forward(d.ax, num_node_list=[5, 6, 4, 1])
    x = 1       # for setting pdb breakpoint

if __name__ == '__main__':
    real_bp()
