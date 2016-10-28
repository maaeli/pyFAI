# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2016 European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" Bayesian evaluation of background for 1D powder diffraction pattern
 
Code according to Sivia and David, J. Appl. Cryst. (2001). 34, 318-324
# Version: 0.1 2012/03/28
# Version: 0.2 2016/10/07: OOP implementation
# Version: 0.3 2016/10/18: Cython parallel implementation
"""

from __future__ import absolute_import, print_function, division

__authors__ = ["Vincent Favre-Nicolin", "Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/10/2016"
__status__ = "development"
__docformat__ = 'restructuredtext'

import cython 
from cython.parallel import prange
import numpy
cimport numpy as cnp
from libc.math cimport log  
from scipy.interpolate import UnivariateSpline


cdef class BackgroundLogLikeliHood:
    cdef: 
        double prefactor, offset, limit1, limit2, c0, c1, c2, c3 

    def __init__(self, double negative_prefactor=1.0, double junction=1.76382127, double delta=1.0):
        """Constructor for the Bayesian likelihood function
        
        This function is composed of 3 parts:
        
        x<<junction: a pure quadratic function x -> x*x 
        maybe with a penalization value for negative x
        
        x>>junction: a pure logarithmic bahviour x -> log(x) - log(junction) + junction*junction
        
        junction-delta < x < junction + delta: a cubic polynomial is fitted to ensure 
        continuity and first derivative continuity. 
        As concavity changes from quadratic to logarithmic, there is no higher order continuity.    
          
        
        :param negative_prefactor: penalization for negative values.
        :param junction: values at which the quadratic function crosses the logarithmic part
        :param delta: size of the junction part.
        """
        assert junction > 0, "Junction value has to be positive"
        if delta < 0:
            delta -= delta
            
        self.limit1 = junction - delta
        self.limit2 = junction + delta
        self.offset = junction * junction - log(junction)
        self.prefactor = negative_prefactor
        
        # calculate the coefficients of the third order polynomial fitting 
        # the values and first order derivatives at the two limits 
        A = numpy.array([[3 * self.limit1 ** 2, 2 * self.limit1, 1, 0], 
                         [3 * self.limit2 ** 2, 2 * self.limit2, 1, 0], 
                         [self.limit1 ** 3, self.limit1 ** 2, self.limit1, 1],
                         [self.limit2 ** 3, self.limit2 ** 2, self.limit2, 1]])
        b = numpy.array([2 * self.limit1, 
                         1. / self.limit2, 
                         self.limit1 ** 2, 
                         log(self.limit2) + self.offset])
        c = numpy.linalg.solve(A, b)
        self.c3, self.c2, self.c1, self.c0 = c
    
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    def summed(self, data, mask=None):
        """Calculate the sum of the LLK function.
        
        :param data: ideally in double[:]
        :param mask: ideally in int8[:]
        :return: sum of log-likelihood over all valid points. 
        """
        cdef:
            int idx, size
            bint do_mask
            double value, llk, total
            double[:] cdata
            cnp.int8_t[:] cmask  
        
        size = data.size
        cdata = numpy.ascontiguousarray(data, numpy.float64).ravel()
        if mask is not None:
            do_mask = True
            assert mask.size == size, "Array size of mask matches"
            cmask = numpy.ascontiguousarray(mask, numpy.int8).ravel() 
        else:
            do_mask = False
        
        total = 0.0 
        for idx in prange(size, nogil=True):
            if do_mask and cmask[idx]:
                continue
            value = cdata[idx]
            llk = self.one_llk(value)
            total += llk
        return total

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    def summed_w(self, obs, guess, weight, mask=None):
        """Calculate the sum of the LLK function.
        
        :param obs: ideally in double[:]
        :param guess: ideally in double[:]
        :param weight: ideally in double[:]
        :param mask: ideally in int8[:]
        :return: sum of log-likelihood over all valid points. 
        """
        cdef:
            int idx, size
            bint do_mask
            double value, llk, total, w
            double[:] cobs, cguess, cweight
            cnp.int8_t[:] cmask  
        
        size = obs.size
        cobs = numpy.ascontiguousarray(obs, numpy.float64).ravel()
        cguess = numpy.ascontiguousarray(guess, numpy.float64).ravel()
        cweight = numpy.ascontiguousarray(weight, numpy.float64).ravel()
        assert guess.size == size, "Array size of guess matches"
        assert weight.size == size, "Array size of weight matches"
        if mask is not None:
            do_mask = True
            assert mask.size == size, "Array size of mask matches"
            cmask = numpy.ascontiguousarray(mask, numpy.int8).ravel() 
        else:
            do_mask = False
        
        total = 0.0 
        for idx in prange(size, nogil=True):
            if do_mask and cmask[idx]:
                continue
            value = (cobs[idx] - cguess[idx]) * cweight[idx]
            llk = self.one_llk(value)
            total += llk
        return total

    @cython.boundscheck(False)
    cdef inline double one_llk(self, double value) nogil:
        cdef:
            double llk, tar, delta
            int lo
        if value <= self.limit1:
            llk = self.prefactor * value * value
        elif value >= self.limit2:
            llk = log(value) + self.offset
        else:  
            llk = self.c0 + value * (self.c1 + value * (self.c2 + self.c3 * value))
        return llk
    
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    def __call__(self, value):
        """Calculate the log-likelihood on a point or a set of points"""
        cdef:             
            int size, idx
            double[:] res, cdata
            
        if "shape" in dir(value):
            cdata = numpy.ascontiguousarray(value, dtype=numpy.float64).ravel()
            size = cdata.size
            res = numpy.zeros(size, numpy.float64)
            for idx in prange(size, nogil=True):
                res[idx] += self.one_llk(cdata[idx])
            return numpy.asarray(res).reshape(value.shape)    
        else:
            return self.one_llk(value)