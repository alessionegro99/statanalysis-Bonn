#!/usr/bin/env python3

import blocksum as bs
import numpy as np

__all__ = ["jackknife_for_primary", "jackknife_for_secondary"]

#***************************
#library functions

def jackknife_for_primary(func, vec_in, block):
  """Jacknife for primary observables.

  Given a numpy vector "vec_in", compute 
  <func(vec_in)> 
  using blocksize "block" for blocking.
  """

  numblocks=int(len(vec_in)/block)
  end =  block * numblocks

  # cut vec_in to have a number of columns multiple of "block" and apply "func" 
  data=func(vec_in[:end])  

  ris=np.mean(data)

  block_sum_data=bs.blocksum(data, block)
  tmp_sums=np.sum(block_sum_data)
  tmp_sums_broad=np.tile(tmp_sums, numblocks) 
  primary_jack=(tmp_sums_broad - block_sum_data)/((numblocks-1)*block) #jackknife samples
  
  err=np.std(primary_jack)*np.sqrt(numblocks-1)

  return ris, err


def jackknife_for_secondary(func2, block, *args):
  """Jacknife for secondary observables.
  
  Every element of *arg is a list of two element of the form
  args[i]=[func_i, vec_i]
  and the final result is 
  func2(<func_0(vec_0)>, ..,<func_n(vec_n)>) 
  with blocksize "block" for blocking.
  """

  # list of primary jackknife samples
  jack_list=[]

  # list of average values of the primary observables
  mean_list=[]

  for arg in args:
    func_l, vec_l = arg

    numblocks=int(len(vec_l)/block)
    end =  block * numblocks

    # cut vec_l to have a number of columns multiple of "block" and applies func_l 
    data=func_l(vec_l[:end])

    tmp_mean=np.mean(data)

    block_sum_data=bs.blocksum(data, block)
    tmp_sum=np.sum(block_sum_data)
    tmp_sum_broad=np.tile(tmp_sum, numblocks) 
    jack_prim=(tmp_sum_broad - block_sum_data)/((numblocks-1)*block) #jackknife sample

    jack_list.append(jack_prim)
    mean_list.append(tmp_mean)

  secondary_jack=func2(jack_list) #jackknife sample
  
  ris_no_bias=numblocks*func2(mean_list)-(numblocks-1)*np.mean(secondary_jack)
  err=np.std(secondary_jack)*np.sqrt(numblocks-1)

  return ris_no_bias, err


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

  size=10000

  # gaussian independent data 
  mu=1.0
  sigma=0.2
  test_noauto=np.random.normal(mu, sigma, size)

  # NO AUTOCORRELATION

  # test for primary
  print("Test for primary observables without autocorrelation")
  print("result must be compatible with %f" % mu)

  ris, err = jackknife_for_primary(id, test_noauto, 1)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()


  # test for secondary
  print("Test for secondary observables without autocorrelation")
  print("result must be compatible with %f" % (sigma*sigma))

  list0=[square, test_noauto]
  list1=[id, test_noauto]
  ris, err = jackknife_for_secondary(susc, 1, list0, list1)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()

  # WITH AUTOCORRELATION

  auto=100

  test_noauto_raw=np.reshape(test_noauto, (1,-1))

  l=[]
  for i in range(0, auto):
    l.append(test_noauto_raw)
  tmp=np.concatenate(tuple(l)).transpose()
  test_auto=tmp.flatten()

  # test for primary
  print("Test for primary observables with autocorrelation")
  print("result must be compatible with %f" % mu)

  ris, err = jackknife_for_primary(id, test_auto, auto)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()

  # test for secondary
  print("Test for secondary observables with autocorrelation")
  print("result must be compatible with %f" % (sigma*sigma))

  list0=[square, test_auto]
  list1=[id, test_auto]
  ris, err = jackknife_for_secondary(susc, auto, list0, list1)

  print("average = %f" % ris)
  print("    err = %f" % err)
  print()
  print("**********************")

