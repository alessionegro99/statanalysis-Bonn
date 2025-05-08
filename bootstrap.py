#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt


import blocksum as bs
import progressbar as pb
import plot

__all__ = ["bootstrap_for_primary", "bootstrap_for_secondary", "blocksize_analysis_primary"]

#***************************
#library functions

def bootstrap_for_primary(func, vec_in, block, samples, seed=None):
  """Bootstrap for primary observables.

  Given a numpy vector "vec_in", compute 
  <func(vec_in)> 
  using blocksize "block" for blocking
  and "samples" resamplings.
  """

  if not isinstance(block, int):
    print("ERROR: blocksize has to be an integer!")
    sys.exit(1)

  if block<1:
    print("ERROR: blocksize has to be positive!")
    sys.exit(1)

  numblocks=int(len(vec_in)/block)
  end =  block * numblocks

  # cut vec_in to have a number of columns multiple of "block" and apply "func" 
  data=func(vec_in[:end])   

  block_sum_data=bs.blocksum(data, block)/np.float64(block)

  # generate bootstrap samples
  aux=len(block_sum_data)
  
  if seed==None:
    bootsample=np.random.choice(block_sum_data, size=(samples, aux), replace=True )
  else:
    np.random.seed(seed)
    bootsample = np.random.choice(block_sum_data, size=(samples, aux), replace=True)
  # sum up the samples
  risboot=np.sum(bootsample, axis=1)/len(block_sum_data)

  ris=np.mean(risboot)
  err=np.std(risboot, ddof=1)
 
  return ris, err


def bootstrap_for_secondary(func2, block, samples, show_progressbar, *args, seed=None):
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

  secondary_samples=np.empty(samples, dtype=np.float64)

  if seed!=None:
    np.random.seed(seed)
  
  for sample in range(samples):
    if show_progressbar==1:
      pb.progress_bar(sample, samples)

    primary_samples=[]

    numblocks=int(len(args[0][1])/block)
    end =  block * numblocks  
    
    resampling = np.random.randint(0,numblocks,size=numblocks)

    for arg in args:
      func_l, vec_l = arg

      # cut vec_in to have a number of columns multiple of "block" and apply "func" 
      data=func_l(vec_l[:end])  

      #block
      block_sum_data=bs.blocksum(data, block)/np.float64(block)

      #sample average
      tmp = np.average([block_sum_data[i] for i in resampling])  

      primary_samples.append(tmp)

    aux=func2(primary_samples)
    secondary_samples[sample]=aux

  ris=np.mean(secondary_samples)
  err=np.std(secondary_samples, ddof=1)

  return ris, err, secondary_samples

def blocksize_analysis_primary(vec_in, samples, block_vec, savefig=0, path=None):      
  """Blocksize analysis for primary observables.
  
  Given a numpy vector "vec_in", 
  plot the standard deviation on the mean
  of "vec_in" as a function of the blocksize.
  Starting from "blocksize_min" to
  "blocksize_max" with steps of size "blocksize_step".
  Uses "samples" bootstrap samples to compute the error.
  "block_vec" is a vector 
  [blocksize_min,blocksize_max,blocksize_step]
  """
  def id(x):
    return x
  
  err = []
  d_err = []
  
  block_range = range(block_vec[0], block_vec[1], block_vec[2])
  
  for block in block_range:
    pb.progress_bar(block, block_vec[1])
    
    numblocks=int(len(vec_in)/block)
    end =  block * numblocks

    data=vec_in[:end]

    block_sum_data=bs.blocksum(data, block)/np.float64(block)
    
    # estimator for the mean of the average for blocksize=block
    err.append(np.std(block_sum_data)/numblocks**0.5)
  
    tmp = []
    
    for sample in range(samples):  
      resampling = np.random.randint(0,numblocks,size=numblocks)
      # compute the mean of the average on each of the bootstrap samples
      tmp.append(np.std(block_sum_data[resampling])/numblocks**0.5)
    
    # the error on the mean of the average is the stdev of the mean of the average
    # computed on the bootstrap samples
    d_err.append(np.std(tmp))
        
  plt.figure(figsize=(16,12))
  plt.errorbar(block_range, err, yerr=d_err,
                fmt='o-', capsize=3, 
                markersize=2, linewidth=0.375,
                color=plot.color_dict[1])
    
  plt.xlabel(r'$K$')
  plt.ylabel(r'$\sigma_{\overline{F(x)}}$', rotation=0)
  plt.title("Standard deviation of the mean as a function of the blocksize.")
  plt.gca().yaxis.set_label_coords(-0.1, 0.5)

  plt.grid(True, which='both', linestyle='--', linewidth=0.25)
  
  if savefig==0:
    plt.show()
  elif savefig==1:
    plt.savefig(f'{path}/blocksize_analysis.png')
  
#***************************
# unit testing

if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  def id(x):
    return x

  def square(x):
    return x*x

  def susc(x):
    return x[0]-x[1]*x[1]

  size=5000
  samples=500
  
  # gaussian independent data 
  mu=1.0
  sigma=0.2
  rng = np.random.default_rng(123)
  test_noauto=rng.normal(mu, sigma, size)
  
  # NO AUTOCORRELATION

  # test for primary
  print("Test for primary observables without autocorrelation")
  print("result must be compatible with %f" % mu)

  ris, err = bootstrap_for_primary(id, test_noauto, 1, samples, seed=None)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()

  # test for secondary
  print("Test for secondary observables without autocorrelation")
  print("result must be compatible with %f" % (sigma*sigma))

  list0=[square, test_noauto]
  list1=[id, test_noauto]
  ris, err = bootstrap_for_secondary(susc, 1, samples, 1, list0, list1, seed=None)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()
  
  # Parameters
  n = 10000           # Length of the vector
  rho = 0.1         # Autocorrelation coefficient (close to 1 for heavy autocorrelation)
  sigma = 1          # Standard deviation of the white noise

  # Generate white noise
  np.random.seed(0)
  noise = np.random.normal(0, sigma, n)

  # Initialize the vector
  x = np.zeros(n)
  for t in range(1, n):
      x[t] = rho * x[t-1] + noise[t]
    
  blocksize_analysis_primary(x, 200, [1, 500, 10])

  print("**********************")

