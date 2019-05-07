#!/usr/bin/env python3

import blocksum as bs
import numpy as np
import progressbar as pb

__all__ = ["bootstrap_for_primary"]

#***************************
#library functions

def bootstrap_for_primary(func, vec_in, block, samples):
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

  block_sum_data=bs.blocksum(data, block)/float(block)

  # generate bootstrap samples
  aux=len(block_sum_data)
  bootsample=np.random.choice(block_sum_data,(samples, aux) )
  
  # sum up the samples
  risboot=np.sum(bootsample, axis=1)/len(block_sum_data)

  ris=np.mean(risboot)
  err=np.std(risboot, ddof=1)
 
  return ris, err



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
  samples=200

  # gaussian independent data 
  mu=1.0
  sigma=0.2
  test_noauto=np.random.normal(mu, sigma, size)

  # NO AUTOCORRELATION

  # test for primary
  print("Test for primary observables without autocorrelation")
  print("result must be compatible with %f" % mu)

  ris, err = bootstrap_for_primary(id, test_noauto, 1, samples)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()


  # WITH AUTOCORRELATION
    
  auto=100

  test_auto=np.empty(0, dtype=float)
  for element in test_noauto:
    tmp=element*np.ones(auto, dtype=float)
    test_auto=np.concatenate((test_auto, tmp))

  # test for primary
  print("Test for primary observables with autocorrelation")
  print("result must be compatible with %f" % mu)

  ris, err = bootstrap_for_primary(id, test_noauto, auto, samples)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()
