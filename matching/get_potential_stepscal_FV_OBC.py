import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np

from scipy.optimize import curve_fit

import concatenate
import bootstrap as boot

## usage
# python3 script.py base_dir therm_drop Nt Ns beta (blocksize)

def readfile(path, *, file_name='Wloop.dat'):
    output = np.loadtxt(f'{path}/analysis/{file_name}', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    return np.column_stack(columns)

base_dir = sys.argv[1]
therm_drop = int(sys.argv[2])
Nt = int(sys.argv[3])
Ns = int(sys.argv[4])
beta = float(sys.argv[5])
if len(sys.argv) > 6:
    blocksize = int(sys.argv[6])
    
input_dir = f'{base_dir}/Ns{Ns}/Nt{Nt}_Ns{Ns}_b{beta:.0f}'

wt_max  = Nt

ws_vec = [i for i in range (1, Ns)]
ws_max = len(ws_vec)

sws_vec = [i**2 + j**2 for i in range(1,Ns) for j in range(1,i+1)]
sws_max = len(sws_vec)

class WilsonLoop:
    def __init__(self, loop_type, wt_max, ws_max, ws_vec=None, ws_plot = None):
        self.loop_type = loop_type
        self.wt_max = wt_max
        self.ws_max = ws_max
        self.ws_vec = ws_vec
        self.ws_plot = ws_plot
        
loops = {
    'planar': WilsonLoop('Wloop', wt_max, ws_max, ws_vec, ws_vec),
    'non_planar': WilsonLoop('sWloop', wt_max, sws_max, sws_vec, np.sqrt(sws_vec))
}

def get_potential (path, loop, *, seed=8220, samples=500, blocksize=200):
    '''
    Compute potential from blocked time-series using bootstrap for error estimation.
    '''
    
    loop_type = loop.loop_type
    wt_max = loop.wt_max
    ws_max = loop.ws_max
        
    data = readfile(path, file_name=f'{loop_type}.dat')
    
    def id(x):
        return x

    wsplot = np.arange(1,ws_max+1)
    
    blocksizes = [blocksize] * len(wsplot) 
                
    V, d_V, V_bs = [], [], []
    
    for ws, blocksize in zip(range(ws_max), blocksizes):
        V_ws, d_V_ws, V_bs_ws = [], [], []
                
        for wt in range(wt_max):
            W = data[:, wt + wt_max*ws]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            _, err, bs = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed, returnsamples=1)

            V_ws.append(potential(np.mean(W)))
            d_V_ws.append(err)
            V_bs_ws.append(bs)
        
        V.append(V_ws)
        d_V.append(d_V_ws)
        V_bs.append(V_bs_ws)
        
    np.save(f'{path}/analysis/{loop_type}_potential_wt', np.array([wsplot, V, d_V, V_bs], dtype=object))
     
if not os.path.isdir(f"{input_dir}/analysis/"):
    os.makedirs(f"{input_dir}/analysis/") 

if not os.path.isfile(f"{input_dir}/analysis/Wloop.dat"):
    concatenate.concatenate(f"{input_dir}/data", therm_drop, f"{input_dir}/analysis/")
    
if not os.path.isfile(f"{input_dir}/analysis/potential_wt.npy"):
    for loop in loops.values():
        get_potential(input_dir, loop)