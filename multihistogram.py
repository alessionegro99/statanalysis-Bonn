#!/usr/bin/env python3

import numpy as np
import scipy.misc as spm

__all__ = ["multihisto_for_primary", "multihisto_for_secondary"]


def _energy(vecdata):
  """
  Energy of a configuration carachterized by "vecdata"

  for instance one could have in 4 dim:
  vecdata = [average plaquette, average Polyakov loop]
  energy  = 6*V*(1-vecdata[0])
  """

  return 6.0*8.0**4.0*(1.0-(vecdata[0]+vecdata[1])/2)


def _newlzeta(lzeta, stuff, speedup):
  """
  The values of the partition functions to be computed are fixed points of this function

  Every element of stuff is a list of elements of the form
  stuff[i]=[param_i, vecdata_i, blockdata_i]
  with vecdata=[column0, column1, .... ]
  """
 
  betas=np.array([element[0] for element in stuff])
  stat=np.array([int(len(np.transpose(element[1]))/element[2]) for element in stuff])

  lzetanew=[]

  for betavalue in betas:
    aux=[]
    for element in stuff:
      end=element[2]*int(len(np.transpose(element[1]))/element[2])
  
      energy=_energy(element[1][:end])

      tmp = np.array([ (-1)*spm.logsumexp(  np.array([ np.log(stat[j])-lzeta[j]+(betavalue-betas[j])*value for j in range(len(betas)) ]) ) for value in energy[0:end:speedup] ])
 
      aux.append(spm.logsumexp(tmp))

    lzetanew.append(spm.logsumexp(np.array(aux)))

  return np.array(lzetanew)


def _computelzeta(stuff):
  """
  Compute the log of the partition functions in a self-consistent way

  Every element of stuff is a list of elements of the form
  stuff[i]=[param_i, vecdata_i, blockdata_i]
  with vecdata=[column0, column1, .... ]
  """

  speedup=int(min(np.array([len(np.transpose(element[1])) for element in stuff]))/10)
  if(speedup<1):
    speedup=1

  lzetaold=np.linspace(1, 2, len(stuff))

  lzetanew=_newlzeta(lzetaold, stuff, speedup)
  check=np.linalg.norm(lzetanew-lzetaold)
  lzetanew-=(lzetanew[0]-1)

  while speedup>1:
    lzetaold=np.copy(lzetanew)
    lzetanew=_newlzeta(lzetaold, stuff, speedup)
    lzetanew-=(lzetanew[0]-1)
    check=np.linalg.norm(lzetanew-lzetaold)

    while check>10e-3:
      lzetaold=np.copy(lzetanew)
      lzetanew=_newlzeta(lzetaold, stuff, speedup)
      lzetanew-=(lzetanew[0]-1)
      check=np.linalg.norm(lzetanew-lzetaold)

      print("computing lzeta: {:.8g} ({:8d})".format(check, speedup), end='\r')

    speedup=int(speedup/2)
    if speedup<1:
      speedup=1

  while check>10e-5:
    lzetaold=np.copy(lzetanew)
    lzetanew=_newlzeta(lzetaold, stuff, speedup)
    lzetanew-=(lzetanew[0]-1)
    check=np.linalg.norm(lzetanew-lzetaold)
    print("computing lzeta: {:.8g} ({:8d})".format(check, speedup), end='\r' )

  return lzetanew


def _jack_for_primary(func, param, lzeta, stuff):
  """
  func is the function of the primaries to be comuted, 
  lzeta is the log of the partition functions 
  and every element of stuff is a list of elements of the form

  stuff[i]=[param_i, vecdata_i, blockdata_i]

  with vecdata=[column0, column1, .... ]
  """

  betas=np.array([element[0] for element in stuff])
  stat=np.array([int(len(np.transpose(element[1]))/element[2]) for element in stuff])
  numblocks=np.sum(stat)

  auxZ=[]
  auxO=[]
  for element in stuff:
    end=element[2]*int(len(np.transpose(element[1]))/element[2])
  
    energy=_energy(element[1])
    obs=func(element[1])

    tmpZ = np.array([ (-1)*spm.logsumexp(  np.array([ np.log(stat[j])-lzeta[j]+(param-betas[j])*value for j in range(len(betas)) ]) ) for value in energy[:end] ])
    tmpO = tmpZ+np.log(obs[:end])
 
    auxZ = np.concatenate(( auxZ, spm.logsumexp( np.reshape(tmpZ,(-1, element[2])), axis=1) ))
    auxO = np.concatenate(( auxO, spm.logsumexp( np.reshape(tmpO,(-1, element[2])), axis=1) ))
 
  sumZ=spm.logsumexp(auxZ)
  sumO=spm.logsumexp(auxO)

  sumZb=np.tile(sumZ, len(auxZ)) 
  sumOb=np.tile(sumO, len(auxO)) 

  jack=np.exp( spm.logsumexp( np.stack((sumOb, auxO), axis=1), b=[1,-1], axis=1) - spm.logsumexp( np.stack((sumZb, auxZ), axis=1), b=[1,-1], axis=1) )

  return np.exp(sumO-sumZ), np.std(jack)*np.sqrt(numblocks-1)


def multihisto_for_primary(param_min, param_max, num_steps, func, *args):
  """
  Compute the primary function func using multihistogram reweighting techniques with 
  weight given by _energy for num_steps values of the control parameter, going from
  param_min to param_max.

  Every element of *args is a list of two element of the form
  args[i]=[param_i, vecdata_i, blockdata_i]
  """

  for arg in args:
    if not isinstance(arg[2], int):
      print("ERROR: blocksize has to be an integer!")
      sys.exit(1)
  
    if arg[2]<1:
      print("ERROR: blocksize has to be positive!")
      sys.exit(1)

  betas=np.array([element[0] for element in args])
  stat=np.array([int(len(np.transpose(element[1]))/element[2]) for element in args])

  print("Values of the control parameters: ", betas)
  print("Statistics: ", stat)
  print("")

  lzeta=_computelzeta(args);

  print("log(zeta) = ", lzeta)
  print("")

  for i in range(num_steps+1):
    param=param_min+i*(param_max-param_min)/num_steps
    ris, err = _jack_for_primary(func, param, lzeta, args) 
    print("{:.8f} {:.12g} {:.12g}".format(param, ris, err) )


def _jack_for_secondary(func2, param, lzeta, stuff, vecfunc):
  """
  func2 is the function to be computed, 
  lzeta is the log of the partition functions 
  and every element of stuff is a list of elements of the form

  stuff[i]=[param_i, vecdata_i, blockdata_i]

  with vecdata=[column0, column1, .... ] and

  vecfunc=[func1a, func1b, func1c..... ]

  where func1a, func1b and so on are the functions of the
  primaries that are needed for func2.
  """

  # list of primary jackknife samples
  jack_list=[]

  # list of average values of the primary observables
  mean_list=[]

  betas=np.array([element[0] for element in stuff])
  stat=np.array([int(len(np.transpose(element[1]))/element[2]) for element in stuff])
  numblocks=np.sum(stat)

  for func1 in vecfunc:
    auxZ=[]
    auxO=[]
    for element in stuff:
      end=element[2]*int(len(np.transpose(element[1]))/element[2])
    
      energy=_energy(element[1])
      obs=func1(element[1])
  
      tmpZ = np.array([ (-1)*spm.logsumexp(  np.array([ np.log(stat[j])-lzeta[j]+(param-betas[j])*value for j in range(len(betas)) ]) ) for value in energy[:end] ])
      tmpO = tmpZ+np.log(obs[:end])
   
      auxZ = np.concatenate(( auxZ, spm.logsumexp( np.reshape(tmpZ,(-1, element[2])), axis=1) ))
      auxO = np.concatenate(( auxO, spm.logsumexp( np.reshape(tmpO,(-1, element[2])), axis=1) ))
   
    sumZ=spm.logsumexp(auxZ)
    sumO=spm.logsumexp(auxO)
  
    sumZb=np.tile(sumZ, len(auxZ)) 
    sumOb=np.tile(sumO, len(auxO)) 
  
    jack_list.append( np.exp( spm.logsumexp( np.stack((sumOb, auxO), axis=1), b=[1,-1], axis=1) - spm.logsumexp( np.stack((sumZb, auxZ), axis=1), b=[1,-1], axis=1) ) )
    mean_list.append( np.exp(sumO-sumZ) )
  
  secondary_jack=func2(jack_list) #jackknife sample
  
  ris_no_bias=numblocks*func2(mean_list) -(numblocks-1)*np.mean(secondary_jack)
  err=np.std(secondary_jack)*np.sqrt(numblocks-1)

  return ris_no_bias, err


def multihisto_for_secondary(param_min, param_max, num_steps, func2, vecfunc, *args):
  """
  Compute the secondary function func2 using multihistogram reweighting techniques 
  with weight given by _energy for num_steps values of the control parameter, going from
  param_min to param_max.

  Every element of *arg is a list of two element of the form
  args[i]=[param_i, vecdata_i, blockdata_i]

  vecfunc=[func1a, func1b, func1c..... ]
  where func1a, func1b and so on are the functions of the
  primaries that are needed for func2.

  """

  for arg in args:
    if not isinstance(arg[2], int):
      print("ERROR: blocksize has to be an integer!")
      sys.exit(1)
  
    if arg[2]<1:
      print("ERROR: blocksize has to be positive!")
      sys.exit(1)

  betas=np.array([element[0] for element in args])
  stat=np.array([int(len(np.transpose(element[1]))/element[2]) for element in args])

  print("Values of the control parameters: ", betas)
  print("Statistics: ", stat)
  print("")

  lzeta=_computelzeta(args);

  print("log(zeta) = ", lzeta)
  print("")

  for i in range(num_steps+1):
    param=param_min+i*(param_max-param_min)/num_steps
    ris, err = _jack_for_secondary(func2, param, lzeta, args, vecfunc) 
    print("{:.8f} {:.12g} {:.12g}".format(param, ris, err) )



#***************************
# unit testing



if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()

  #indata=np.loadtxt("dati_2.40.dat", skiprows=0, dtype=np.float)
  #data240=np.transpose(indata)     #column ordered

  #indata=np.loadtxt("dati_2.42.dat", skiprows=0, dtype=np.float)
  #data242=np.transpose(indata)     #column ordered

  #indata=np.loadtxt("dati_2.44.dat", skiprows=0, dtype=np.float)
  #data244=np.transpose(indata)     #column ordered

  #def plaq(vec):
  #  return (vec[0]+vec[1])/2

  #def squareplaq(vec):
  #  return (vec[0]+vec[1])*(vec[0]+vec[1])/4

  #def plaqsusc(v):
  #  return v[0]-v[1]*v[1]

  #multihisto_for_primary(2.40, 2.44, 200, plaq, [2.40, data240, 1], [2.42, data242, 1], [2.44, data244, 1])

  #multihisto_for_secondary(2.40, 2.44, 20, plaqsusc, [squareplaq, plaq], [2.40, data240, 10], [2.42, data242, 10], [2.44, data244, 10])

  print("**********************")

