import pickle
from numpy.random import random as rnd
import os

high = 0.05
low = -high
N = 40

x0 = low + (high-low)*rnd((N,1))
x_dot0 = low + (high-low)*rnd((N,1))
th0 = low + (high-low)*rnd((N,1))
th_dot0 = low + (high-low)*rnd((N,1))
initial_pos = [x0, x_dot0, th0, th_dot0]


# Current Folder path
cur_folder = os.path.dirname(os.path.realpath(__file__))

with open(cur_folder + '/storage', 'wb') as fp:
        pickle.dump(initial_pos, fp)