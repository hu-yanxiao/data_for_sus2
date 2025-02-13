#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from glob import glob
import os
import sys
import datetime
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from scipy.signal import correlate

def acf(x, unbiased=True):
    """ auto correlation for x. if unbiased use n-k as denominators, otherwise n"""
    N = len(x)
    cN=int(100/0.001/10)
    #fvi = fft(x, n=2 * N)
    d = N - np.arange(N) if unbiased else N
    result = correlate(x, x, mode='full',method='fft')[-N:]/d
    #for sn in range(0,N-cN):
    #    result=np.zeros(cN)
    #    for cn in range(0,cN):
    #        result[cn]=(x[sn+cn]*x[sn])/(N-cN)
    return result


def ccf(x, y, unbiased=True):
    """ cross correlation for x and y. if unbiased use n-k as denominators, otherwise n"""
    N = len(x)
    d = N - np.arange(N) if unbiased else N
    fx = fft(x, n=2 * N)
    fy = fft(y, n=2 * N)
    result = np.real(ifft(fx * np.conjugate(fy))[:N] / d)
    return result


def find_fn():
    return sorted(glob("heatfulx-*.dat"))


def main(debug=False, overwrite=True, max_step=-1,dn=1):
    if debug:
        print("DEBUG MODE")
        max_step = 10000

    fns = find_fn()
    print(fns)

    for f1 in fns:
        print("Reading file:\n{}\n".format(f1))
        hcacf_fn = os.path.splitext(f1)[0].replace("heatfulx",
                                                   "hcacf") + ".feather"
        if not overwrite and os.path.exists(hcacf_fn):
            print("Input file:{}\tfile exists: {}, do nothing".format(
                fn, hcacf_fn))
            continue
        dd = pd.read_csv(f1, sep=" ")
        hcacf = pd.DataFrame()

        #hcacf['JJ'] = (acf(dd.Jx[:max_step]) + acf(dd.Jy[:max_step]) +
        #               acf(dd.Jz[:max_step])) / 3.0
        hcacf['JJ'] = (acf(dd.Jx[:max_step:dn]-0*dd.Jx1[:max_step:dn]) + acf(dd.Jy[:max_step:dn]-0*dd.Jy1[:max_step:dn]) +
                acf(dd.Jz[:max_step:dn]-0*dd.Jz1[:max_step:dn])) / 3.0

        hcacf['JJ2'] = (acf(dd.Jx1[:max_step]) +
                         acf(dd.Jy1[:max_step]) +
                         acf(dd.Jz1[:max_step])) / 3.0
        hcacf['JJ3'] = (acf(dd.Jx[:max_step:dn]-1*dd.Jx1[:max_step:dn]) + acf(dd.Jy[:max_step:dn]-1*dd.Jy1[:max_step:dn]) +
                acf(dd.Jz[:max_step:dn]-1*dd.Jz1[:max_step:dn])) / 3.0

        #hcacf['JJ2'] = acf(dd.Jz[:max_step] - dd.Jz1[:max_step])

        hcacf.to_feather(hcacf_fn)

        print("Output file: {}\n".format(hcacf_fn))


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    main()
