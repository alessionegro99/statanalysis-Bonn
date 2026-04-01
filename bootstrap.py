#!/usr/bin/env python3

import sys
from typing import Callable

import numpy as np


def blocksum(vec_in: np.ndarray, block: int) -> np.ndarray:
    """Return block sums of a 1D array."""
    vec = np.asarray(vec_in)
    n_blocks = len(vec) // block
    trimmed = vec[: n_blocks * block]
    return trimmed.reshape(n_blocks, block).sum(axis=1)


def bootstrap_for_primary(func, vec_in, block, samples, seed=None, returnsamples=False):
    """Bootstrap for primary observables.

    Given a numpy vector "vec_in", compute
    <func(vec_in)>
    using blocksize "block" for blocking
    and "samples" resamplings.
    """

    if not isinstance(block, int):
        print("ERROR: blocksize has to be an integer!")
        sys.exit(1)

    if block < 1:
        print("ERROR: blocksize has to be positive!")
        sys.exit(1)

    numblocks = int(len(vec_in) / block)
    end = block * numblocks

    # cut vec_in to have a number of columns multiple of "block" and apply "func"
    data = func(vec_in[:end])

    # get the average on the blocks
    block_sum_data = blocksum(data, block) / np.float64(block)

    # generate bootstrap samples
    aux = len(block_sum_data)

    if seed is None:
        bootsample = np.random.choice(block_sum_data, size=(samples, aux), replace=True)
    else:
        np.random.seed(seed)
        bootsample = np.random.choice(block_sum_data, size=(samples, aux), replace=True)
    # sum up the samples
    risboot = np.sum(bootsample, axis=1) / len(block_sum_data)

    ris = np.mean(risboot)
    err = np.std(risboot, ddof=1)

    if returnsamples:
        return ris, err, risboot

    return ris, err


def bootstrap_for_secondary(
    func2: Callable[[list[float]], float],
    block: int,
    samples: int,
    *observables: tuple[Callable[[np.ndarray], np.ndarray], np.ndarray],
    seed: int | None = None,
    returnsamples: bool = False,
):
    """
    Bootstrap for secondary observables.

    Each element of `observables` must be a pair `(func_i, vec_i)`.
    For each bootstrap replica, the function computes

        func2([ <func_0(vec_0)>, ..., <func_n(vec_n)> ])

    where each primary observable is first blocked with block size `block`,
    then resampled at the block level.
    """
    if not isinstance(block, int):
        raise TypeError("block must be an integer")
    if block < 1:
        raise ValueError("block must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not observables:
        raise ValueError("at least one observable must be provided")

    rng = np.random.default_rng(seed)
    secondary_samples = np.empty(samples, dtype=np.float64)

    n = len(observables[0][1])
    numblocks = n // block
    if numblocks < 2:
        raise ValueError("Not enough data for this block size")

    end = block * numblocks

    for sample in range(samples):
        resampling = rng.integers(0, numblocks, size=numblocks)
        primary_samples: list[float] = []

        for func_i, vec_i in observables:
            data = func_i(np.asarray(vec_i[:end]))
            blocked = blocksum(data, block) / float(block)
            primary_samples.append(np.mean(blocked[resampling]))

        secondary_samples[sample] = func2(primary_samples)

    mean = np.mean(secondary_samples)
    err = np.std(secondary_samples, ddof=1)

    if returnsamples:
        return mean, err, secondary_samples
    return mean, err


# ***************************
# unit testing

if __name__ == "__main__":
    print("**********************")
    print("UNIT TESTING")
    print()

    def id(x):
        return x

    def square(x):
        return x * x

    def susc(x):
        return x[0] - x[1] * x[1]

    size = 5000
    samples = 500

    # gaussian independent data
    mu = 1.0
    sigma = 0.2
    rng = np.random.default_rng(123)
    test_noauto = rng.normal(mu, sigma, size)

    # NO AUTOCORRELATION

    # test for primary
    print("Test for primary observables without autocorrelation")
    print("result must be compatible with %f" % mu)

    ris, err = bootstrap_for_primary(id, test_noauto, 1, samples, seed=None)

    print("average = %f" % ris)
    print("    err = %f" % err)
    print("    std = %f" % (np.std(test_noauto) / len(test_noauto) ** 0.5))
    print()

    # test for secondary
    print("Test for secondary observables without autocorrelation")
    print("result must be compatible with %f" % (sigma * sigma))

    list0 = [square, test_noauto]
    list1 = [id, test_noauto]
    ris, err = bootstrap_for_secondary(susc, 1, samples, 1, list0, list1, seed=None)

    print("average = %f" % ris)
    print("    err = %f" % err)
    print()

    print("**********************")
