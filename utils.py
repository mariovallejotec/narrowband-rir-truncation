# -*- coding: utf-8 -*-
"""Helpers shared by both pipelines: time alignment and unimodal regression.

Inherited from the RIR-cropping code base (K. Prawda).
"""
import numpy as np
from scipy import signal
from scipy.optimize import isotonic_regression


def time_align_rirs(upsampled_rir, referenceID):
    """Time-align upsampled RIRs with the help of cross-correlation.

    Parameters
    ----------
    upsampled_rir : 2-D array (n_samples, n_rirs)
        Upsampled RIRs, one per column.
    referenceID : int
        Column index of the reference RIR; every later column is shifted to
        match this one.

    Returns
    -------
    upsampled_rir : 2-D array
        The same array with the non-reference columns aligned (in place).
    """
    num_repeat = upsampled_rir.shape[-1]

    for rep in range(referenceID + 1, num_repeat):
        # calculate the cross-correlation of upsampled sweeps
        corr = signal.correlate(upsampled_rir[:, referenceID], upsampled_rir[:, rep])
        # get the time lag as well
        lags = signal.correlation_lags(len(upsampled_rir[:, referenceID]), len(upsampled_rir[:, rep]))
        # find where the zero lag is - the lag is symmetric on both sides
        lag_ind = np.nonzero(lags == 0)
        zero_lag = lag_ind[0][0]  # the position of zero lag

        # find the lag of the max correlation - this will define the amount of samples we need to shift
        max_lag = np.argmax(corr)
        sign = np.sign(max_lag - zero_lag)  # sign will tell us whether to shift to left or right
        shift = np.abs(max_lag - zero_lag) + 1  # needs to be shifted by 1 because of indexing through zero

        # shift the IR
        shifted = np.roll(upsampled_rir[:, rep], sign * shift)
        upsampled_rir[:, rep] = shifted

    return upsampled_rir


def unimodal_regression(sig):
    """Smooth a signal with unimodal regression: split it at its maximum into
    an increasing and a decreasing region and apply isotonic regression to
    each of them.

    Parameters
    ----------
    sig : 1-D array
        Signal vector.

    Returns
    -------
    sig_unimodal : 1-D array
        Signal smoothed using unimodal regression.
    ind : int
        Index of the maximum of the signal, where the increasing regression
        turns into the decreasing one.
    """
    # we need to separate the increasing and decreasing parts of the signal:
    # find the maximum of the signal - we assume there are clearly separable
    # increasing and decreasing sections
    ind = int(np.argmax(sig))

    # decreasing regression
    result = isotonic_regression(sig[ind:], increasing=False)
    sig_isReg_dec = result.x

    if ind > 0:
        # increasing regression
        result = isotonic_regression(sig[:ind], increasing=True)
        sig_isReg_inc = result.x
        sig_unimodal = np.concatenate((sig_isReg_inc, sig_isReg_dec))
    else:
        sig_unimodal = sig_isReg_dec

    return sig_unimodal, ind
