# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "10/07/2017"
__copyright__ = "2011-2015, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy
from libc.math cimport floor, ceil
from cython cimport floating
from cython.parallel import prange

import logging
logger = logging.getLogger("pyFAI.ext.bilinear")

from ..decorators import timeit


cdef class Bilinear:
    """Bilinear interpolator for finding max.

    Instance attribute defined in pxd file
    """
#     cdef:
#         readonly float[:, ::1] data
#         readonly float maxi, mini
#         readonly size_t width, height
# 
#     cpdef size_t cp_local_maxi(self, size_t)
#     cdef size_t c_local_maxi(self, size_t) nogil

    def __cinit__(self, data not None):
        assert data.ndim == 2
        self.height = data.shape[0]
        self.width = data.shape[1]
        self.maxi = data.max()
        self.mini = data.min()
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)

    def __dealloc__(self):
        self.data = None

    def f_cy(self, x):
        """
        Function -f((y,x)) where f is a continuous function 
        (y,x) are pixel coordinates
        pixels outside the image are given an arbitrary high value to help the minimizer  

        :param x: 2-tuple of float
        :return: Interpolated negative signal from the image 
        (negative for using minimizer to search for peaks)
        """
        cdef:
            float d0 = x[0]
            float d1 = x[1]
        if d0 < 0:
            res = self.mini + d0
        elif d1 < 0:
            res = self.mini + d1
        elif d0 > (self.height - 1):
            res = self.mini - d0 + self.height - 1
        elif d1 > self.width - 1:
            res = self.mini - d1 + self.width - 1
        else:
            res = self._f_cy(d0, d1)
        return -res 
    
    def __call__(self, x):
        "Function f((y,x)) where f is a continuous function "
        cdef:
            float d0 = x[0]
            float d1 = x[1]
        return self._f_cy(d0, d1)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float _f_cy(self, floating d0, floating d1) nogil:
        """
        Function f((y,x)) where f is a continuous function (y,x) are pixel coordinates

        :param x: 2-tuple of float
        :return: Interpolated signal from the image
        """

        cdef:
            int i0, i1, j0, j1
            float x0, x1, y0, y1, res
        if d0 < 0:
            d0 = 0
        elif d1 < 0:
            d1 = 0
        elif d0 > (self.height - 1):
            d0 = self.height - 1
        elif d1 > self.width - 1:
            d1 = self.width - 1
        x0 = floor(d0)
        x1 = ceil(d0)
        y0 = floor(d1)
        y1 = ceil(d1)
        i0 = < int > x0
        i1 = < int > x1
        j0 = < int > y0
        j1 = < int > y1
        if (i0 == i1) and (j0 == j1):
            res = self.data[i0, j0]
        elif i0 == i1:
            res = (self.data[i0, j0] * (y1 - d1)) + (self.data[i0, j1] * (d1 - y0))
        elif j0 == j1:
            res = (self.data[i0, j0] * (x1 - d0)) + (self.data[i1, j0] * (d0 - x0))
        else:
            res = (self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                + (self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                + (self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                + (self.data[i1, j1] * (d0 - x0) * (d1 - y0))
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def local_maxi(self, x):
        """
        Return the local maximum with sub-pixel refinement.

        Sub-pixel refinement:
        Second order Taylor expansion of the function; first derivative is null

        .. math:: delta = x-i = -Inverse[Hessian].gradient

        If Hessian is singular or :math:`|delta|>1`: use a center of mass.

        :param x: 2-tuple of integers
        :param w: half with of the window: 1 or 2 are advised
        :return: 2-tuple of float with the nearest local maximum
        """
        cdef:
            int res, current0, current1
            int i0, i1
            float tmp, sum0 = 0, sum1 = 0, sum = 0
            float a00, a01, a02, a10, a11, a12, a20, a21, a22
            float d00, d11, d01, denom, delta0, delta1

        res = self.c_local_maxi(round(x[0]) * self.width + round(x[1]))

        current0 = res // self.width
        current1 = res % self.width
        if (current0 > 0) and (current0 < self.height - 1) and (current1 > 0) and (current1 < self.width - 1):
            # Use second order polynomial Taylor expansion
            a00 = self.data[current0 - 1, current1 - 1]
            a01 = self.data[current0 - 1, current1    ]
            a02 = self.data[current0 - 1, current1 + 1]
            a10 = self.data[current0    , current1 - 1]
            a11 = self.data[current0    , current1    ]
            a12 = self.data[current0    , current1 + 1]
            a20 = self.data[current0 + 1, current1 - 1]
            a21 = self.data[current0 + 1, current1    ]
            a22 = self.data[current0 + 1, current1 - 1]
            d00 = a12 - 2.0 * a11 + a10
            d11 = a21 - 2.0 * a11 + a01
            d01 = (a00 - a02 - a20 + a22) / 4.0
            denom = 2.0 * (d00 * d11 - d01 * d01)
            if abs(denom) < 1e-10:
                logger.debug("Singular determinant, Hessian undefined")
            else:
                delta0 = ((a12 - a10) * d01 + (a01 - a21) * d11) / denom
                delta1 = ((a10 - a12) * d00 + (a21 - a01) * d01) / denom
                if abs(delta0) <= 1.0 and abs(delta1) <= 1.0:
                    # Result is OK if lower than 0.5.
                    return (delta0 + float(current0), delta1 + float(current1))
                else:
                    logger.debug("Failed to find root using second order expansion")
            # refinement of the position by a simple center of mass of the last valid region used
            for i0 in range(current0 - 1, current0 + 2):
                for i1 in range(current1 - 1, current1 + 2):
                    tmp = self.data[i0, i1]
                    sum0 += tmp * i0
                    sum1 += tmp * i1
                    sum += tmp
            if sum > 0:
                return (sum0 / sum, sum1 / sum)

        return (float(current0), float(current1))

    cpdef size_t cp_local_maxi(self, size_t x):
        return self.c_local_maxi(x)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef size_t c_local_maxi(self, size_t x) nogil:
        """
        Return the local maximum ... without sub-pixel refinement

        :param x: start index
        :param w: half with of the window: 1 or 2 are advised
        :return: local maximum index
        """
        cdef:
            int current0 = x // self.width
            int current1 = x % self.width
            int i0, i1, start0, stop0, start1, stop1, new0, new1
            float tmp, value, old_value

        value = self.data[current0, current1]
        old_value = value - 1.0
        new0, new1 = current0, current1

        while value > old_value:
            old_value = value
            start0 = max(0, current0 - 1)
            stop0 = min(self.height, current0 + 2)
            start1 = max(0, current1 - 1)
            stop1 = min(self.width, current1 + 2)
            for i0 in range(start0, stop0):
                for i1 in range(start1, stop1):
                    tmp = self.data[i0, i1]
                    if tmp > value:
                        new0, new1 = i0, i1
                        value = tmp
            current0, current1 = new0, new1
        return self.width * current0 + current1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_cartesian_positions(floating[::1] d1, floating[::1] d2,
                             float[:, :, :, ::1] pos,
                             bint is_flat=True):
    """
    Calculate the Cartesian position for array of position (d1, d2)
    with pixel coordinated stored in array pos.
    This is bilinear interpolation.

    :param d1: position in dim1
    :param d2: position in dim2
    :param pos: array with position of pixels corners
    :return: 3-tuple of position.
    """
    cdef:
        int i, p1, p2, dim1, dim2, size = d1.size
        float delta1, delta2, f1, f2, A0, A1, A2, B0, B1, B2, C1, C0, C2, D0, D1, D2
        numpy.ndarray[numpy.float32_t, ndim = 1] out1 = numpy.zeros(size, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] out2 = numpy.zeros(size, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] out3
    if not is_flat:
        out3 = numpy.zeros(size, dtype=numpy.float32)
    dim1 = pos.shape[0]
    dim2 = pos.shape[1]
    assert size == d2.size, "d2.size == size"

    for i in prange(size, nogil=True, schedule="static"):
        f1 = floor(d1[i])
        f2 = floor(d2[i])

        p1 = <int> f1
        p2 = <int> f2

        delta1 = d1[i] - f1
        delta2 = d2[i] - f2

        if p1 < 0:
            with gil:
                print("f1= %s" % f1)

        if p1 < 0:
            with gil:
                print("f2= %s" % f2)

        if p1 >= dim1:
            if p1 > dim1:
                with gil:
                    print("d1= %s, f1=%s, p1=%s, delta1=%s" % (d1[i], f1, p1, delta1))
            p1 = dim1 - 1
            delta1 = d1[i] - p1

        if p2 >= dim2:
            if p2 > dim2:
                with gil:
                    print("d2= %s, f2=%s, p2=%s, delta2=%s" % (d2[i], f2, p2, delta2))
            p2 = dim2 - 1
            delta2 = d2[i] - p2

        A1 = pos[p1, p2, 0, 1]
        A2 = pos[p1, p2, 0, 2]
        B1 = pos[p1, p2, 1, 1]
        B2 = pos[p1, p2, 1, 2]
        C1 = pos[p1, p2, 2, 1]
        C2 = pos[p1, p2, 2, 2]
        D1 = pos[p1, p2, 3, 1]
        D2 = pos[p1, p2, 3, 2]
        if not is_flat:
            A0 = pos[p1, p2, 0, 0]
            B0 = pos[p1, p2, 1, 0]
            C0 = pos[p1, p2, 2, 0]
            D0 = pos[p1, p2, 3, 0]
            out3[i] += A0 * (1.0 - delta1) * (1.0 - delta2) \
                + B0 * delta1 * (1.0 - delta2) \
                + C0 * delta1 * delta2 \
                + D0 * (1.0 - delta1) * delta2

        # A and D are on the same:  dim1 (Y)
        # A and B are on the same:  dim2 (X)
        # nota: += is needed as well as numpy.zero because of prange: avoid reduction
        out1[i] += A1 * (1.0 - delta1) * (1.0 - delta2) \
            + B1 * delta1 * (1.0 - delta2) \
            + C1 * delta1 * delta2 \
            + D1 * (1.0 - delta1) * delta2
        out2[i] += A2 * (1.0 - delta1) * (1.0 - delta2) \
            + B2 * delta1 * (1.0 - delta2) \
            + C2 * delta1 * delta2 \
            + D2 * (1.0 - delta1) * delta2
    if is_flat:
        return out1, out2, None
    else:
        return out1, out2, out3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convert_corner_2D_to_4D(int ndim,
                            floating[:, ::1] d1 not None,
                            floating[:, ::1] d2 not None,
                            floating[:, ::1] d3=None):
    """
    Convert 2 (or 3) arrays of corner position into a 4D array of pixel corner coordinates

    :param ndim: 2d or 3D output
    :param d1: 2D position in dim1 (shape +1)
    :param d2: 2D position in dim2 (shape +1)
    :param d3: 2D position in dim3 (z) (shape +1)
    :return: pos 4D array with position of pixels corners
    """
    cdef int shape0, shape1, i, j
    #  edges position are n+1 compared to number of pixels
    shape0 = d1.shape[0] - 1
    shape1 = d2.shape[1] - 1
    assert d1.shape[0] == d2.shape[0], "d1.shape[0] == d2.shape[0]"
    assert d1.shape[1] == d2.shape[1], "d1.shape[1] == d2.shape[1]"
    cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos = numpy.zeros((shape0, shape1, 4, ndim), dtype=numpy.float32)
    for i in prange(shape0, nogil=True, schedule="static"):
        for j in range(shape1):
            pos[i, j, 0, ndim - 2] += d1[i, j]
            pos[i, j, 0, ndim - 1] += d2[i, j]
            pos[i, j, 1, ndim - 2] += d1[i + 1, j]
            pos[i, j, 1, ndim - 1] += d2[i + 1, j]
            pos[i, j, 2, ndim - 2] += d1[i + 1, j + 1]
            pos[i, j, 2, ndim - 1] += d2[i + 1, j + 1]
            pos[i, j, 3, ndim - 2] += d1[i, j + 1]
            pos[i, j, 3, ndim - 1] += d2[i, j + 1]
    if (d3 is not None) and (ndim == 3):
        assert d1.shape[0] == d3.shape[0], "d1.shape[0] == d3.shape[0]"
        assert d1.shape[1] == d3.shape[1], "d1.shape[1] == d3.shape[1]"
        for i in prange(shape0, nogil=True, schedule="static"):
            for j in range(shape1):
                pos[i, j, 0, 0] += d3[i, j]
                pos[i, j, 1, 0] += d3[i + 1, j]
                pos[i, j, 2, 0] += d3[i + 1, j + 1]
                pos[i, j, 3, 0] += d3[i, j + 1]
    return pos
