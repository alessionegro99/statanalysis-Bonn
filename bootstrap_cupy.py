#!/usr/bin/env python3

import sys

import cupy as cp

import progressbar as pb

__all__ = ["blocksum", "bootstrap_for_secondary"]

def blocksum(vec_in, block):
  """Block sums of a vector.

  Given a numpy vector "vec_in", it returns the vector of block sums using
  blocksize "block" for blocking. 
  The returned vector has length int(len(vec_in)/block)
  """

  end =  block * int(len(vec_in)/block)
  ris=cp.sum(cp.reshape(vec_in[:end],(-1, block)), axis=1)
  return ris

def bootstrap_for_secondary(func2, block, samples, *args, seed=None, returnsamples=0):
  """Bootstrap for secondary observables.
  
  Every element of *arg is a list of two element of the form
  args[i]=[func_i, vec_i]
  and the final result is 
  func2(<func_0(vec_0)>, ..,<func_n(vec_n)>) 
  with blocksize "block" for blocking
  and "samples" resampling
  show_progressbar: if =1 show the progressbar
  seed: if != 0 the rng is seeded with seed
  """
  if not isinstance(block, int):
    print("ERROR: blocksize has to be an integer!")
    sys.exit(1)

  if block<1:
    print("ERROR: blocksize has to be positive!")
    sys.exit(1)

  secondary_samples=cp.empty(samples, dtype=cp.float64)

  if seed!=None:
    cp.random.seed(seed)
  
  for sample in range(samples):
    primary_samples=[]

    numblocks=int(len(args[0][1])/block)
    end =  block * numblocks  
    
    resampling = cp.random.randint(0,numblocks,size=numblocks)

    for arg in args:
      func_l, vec_l = arg

      # cut vec_in to have a number of columns multiple of "block" and apply "func" 
      data = cp.asarray(func_l(vec_l[:end]))

      #block
      block_sum_data=blocksum(data, block)/cp.float64(block)

      #sample average
      tmp = cp.mean(block_sum_data[resampling])
      
      primary_samples.append(tmp)

    aux=func2(primary_samples)
    secondary_samples[sample]=aux

  ris=cp.mean(secondary_samples).item()
  err=cp.std(secondary_samples, ddof=1).item()

  if returnsamples==1:
    return ris, err, secondary_samples.get()
  else:
    return ris, err

vec1 = cp.random.randn(1000)

