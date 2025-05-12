import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import plot
import progressbar as pb
import bootstrap as boot
import regression as reg

def readfile(path, index):
    output = np.loadtxt(f'{path}/data/dati_{index}.dat', skiprows=0)
    columns = [output[:, i] for i in range(output.shape[1])]

    # acc_rate mag energy
    
    return np.column_stack(columns)