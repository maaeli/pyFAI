# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2016 European Synchrotron Radiation Facility, France
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

"""Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done on the pixel's bounding box similar to fit2D
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "11/07/2017"
__status__ = "stable"
__license__ = "MIT"

cimport numpy
cimport cython
from regrid_common cimport get_bin_number, fabs, EPS32, pi 
import numpy
import logging
logger = logging.getLogger(__name__)
from . import sparse_utils
from .sparse_utils cimport ArrayBuilder
from isnan cimport isnan, isfinite

# define some common types
ctypedef float position_t
ctypedef float data_t
ctypedef double sum_t
position_d = numpy.float32
data_d = numpy.float32
sum_d = numpy.float64


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def histoBBox1d(weights,
                pos0,
                delta_pos0,
                pos1=None,
                delta_pos1=None,
                size_t bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                empty=None,
                double normalization_factor=1.0):

    """
    Calculates histogram of pos0 (tth) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D

    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value

    :return: 2theta, I, weighted histogram, unweighted histogram
    """
    cdef size_t  size = weights.size
    assert pos0.size == size, "pos0.size == size"
    assert delta_pos0.size == size, "delta_pos0.size == size"
    assert bins > 1, "at lease one bin"
    cdef:
        ssize_t   bin0_max, bin0_min
        float data, delta, deltaR, deltaL, deltaA, epsilon = 1e-10, cdummy = 0, ddummy = 0
        float pos0_min=0, pos0_max=0, pos0_maxin=0, pos1_min=0, pos1_max=0, pos1_maxin=0, min0=0, max0=0, fbin0_min=0, fbin0_max=0
        bint check_pos1=False, check_mask=False, check_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
        sum_t[::1] out_data, out_count, out_merge
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        position_t[::1] cpos0, dpos0, cpos1, dpos1, cpos0_lower, cpos0_upper
        numpy.int8_t[:] cmask
        data_t[:] cflat, cdark, cpolarization, csolidangle
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)
    
    out_data = numpy.zeros(bins, dtype=sum_d)
    out_count = numpy.zeros(bins, dtype=sum_d)
    out_merge = numpy.zeros(bins, dtype=sum_d)

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        ddummy = 0.0
    else:
        check_dummy = False
        cdummy = empty or 0.0
        ddummy = 0.0
    if dark is not None:
        assert dark.size == size, "dark current array size"
        do_dark = True
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=data_d)

    cpos0_lower = numpy.zeros(size, dtype=numpy.float32)
    cpos0_upper = numpy.zeros(size, dtype=numpy.float32)
    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    with nogil:
        for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue
            min0 = cpos0[idx] - dpos0[idx]
            max0 = cpos0[idx] + dpos0[idx]
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max
    pos0_min = max(0.0, pos0_min)
    pos0_max = pos0_maxin * EPS32

    if pos1Range is not None and len(pos1Range) > 1:
        assert pos1.size == size, "pos1.size == size"
        assert delta_pos1.size == size, "delta_pos1.size == size"
        check_pos1 = 1
        cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float32)
        dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=numpy.float32)
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        pos1_max = pos1_maxin * EPS32

    delta = (pos0_max - pos0_min) / (<float> (bins))
    outPos = numpy.linspace(pos0_min + 0.5 * delta, pos0_maxin - 0.5 * delta, bins)
    with nogil:
        for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue
            data = cdata[idx]
            if not isfinite(data):
                continue 
            if check_dummy and (fabs(data - cdummy) <= ddummy):
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]

            if check_pos1 and (((cpos1[idx] + dpos1[idx]) < pos1_min) or ((cpos1[idx] - dpos1[idx]) > pos1_max)):
                    continue

            fbin0_min = get_bin_number(min0, pos0_min, delta)
            fbin0_max = get_bin_number(max0, pos0_min, delta)
            if (fbin0_max < 0) or (fbin0_min >= bins):
                continue
            if fbin0_max >= bins:
                bin0_max = bins - 1
            else:
                bin0_max = < ssize_t > fbin0_max
            if fbin0_min < 0:
                bin0_min = 0
            else:
                bin0_min = < ssize_t > fbin0_min

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                out_count[bin0_min] += 1.0
                out_data[bin0_min] += data

            else:
                # we have pixel spliting.
                deltaA = 1.0 / (fbin0_max - fbin0_min)

                deltaL = < float > (bin0_min + 1) - fbin0_min
                deltaR = fbin0_max - (< float > bin0_max)

                out_count[bin0_min] += (deltaA * deltaL)
                out_data[bin0_min] += (data * deltaA * deltaL)

                out_count[bin0_max] += (deltaA * deltaR)
                out_data[bin0_max] += (data * deltaA * deltaR)

                if bin0_min + 1 < bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        out_count[i] += deltaA
                        out_data[i] += (data * deltaA)

        for i in range(bins):
                if out_count[i] > epsilon:
                    out_merge[i] = < float > (out_data[i] / out_count[i] / normalization_factor)
                else:
                    out_merge[i] = cdummy

    return outPos, numpy.asarray(out_merge), numpy.asarray(out_data), numpy.asarray(out_count)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def histoBBox2d(weights,
                pos0,
                delta_pos0,
                pos1,
                delta_pos1,
                bins=(100, 36),
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                bint allow_pos0_neg=0,
                bint chiDiscAtPi=1,
                empty=0.0,
                double normalization_factor=1.0):
    """
    Calculate 2D histogram of pos0(tth),pos1(chi) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins (tth=100, chi=36 by default)
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value


    :return: I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef:
        ssize_t bins0, bins1, i, j, idx, bin0_max, bin0_min, bin1_max, bin1_min
        size_t size = weights.size
    assert pos0.size == size, "pos0.size == size"
    assert pos1.size == size, "pos1.size == size"
    assert delta_pos0.size == size, "delta_pos0.size == size"
    assert delta_pos1.size == size, "delta_pos1.size == size"
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = bins
    bins0 = max(bins0, 1)
    bins1 = max(bins1, 1)
    cdef:
        data_t[::1] cdata, cflat, cdark, cpolarization, csolidangle 
        position_t[::1] cpos0, dpos0, cpos1, dpos1, cpos0_upper, cpos0_lower, cpos1_upper, cpos1_lower
        sum_t[:, ::1] out_data, out_count
        data_t[:, ::1] out_merge
        numpy.int8_t[::1] cmask
        position_t c0, c1, d0, d1
        position_t min0, max0, min1, max1, deltaR, deltaL, deltaU, deltaD, deltaA, delta0, delta1
        position_t pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin
        position_t fbin0_min, fbin0_max, fbin1_min, fbin1_max, 
        data_t data, epsilon = 1e-10, cdummy, ddummy
        bint check_mask=False, check_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False

    cdata = numpy.ascontiguousarray(weights.ravel(), dtype=position_d)
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
    cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
    dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=position_d)
    cpos0_upper = numpy.empty(size, dtype=position_d)
    cpos0_lower = numpy.empty(size, dtype=position_d)
    cpos1_upper = numpy.empty(size, dtype=position_d)
    cpos1_lower = numpy.empty(size, dtype=position_d)
    
    out_data = numpy.zeros((bins0, bins1), dtype=sum_d)
    out_count = numpy.zeros((bins0, bins1), dtype=sum_d)
    out_merge = numpy.zeros((bins0, bins1), dtype=data_d)

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    if (dummy is not None) and delta_dummy is not None:
        check_dummy = True
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
    else:
        cdummy = float(empty)

    if dark is not None:
        assert dark.size == size, "dark current array size"
        do_dark = True
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float32)

    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    pos1_min = cpos1[0]
    pos1_max = cpos1[0]

    with nogil:
        for idx in range(size):
            if (check_mask and cmask[idx]):
                continue
            c0 = cpos0[idx]
            d0 = dpos0[idx]
            min0 = c0 - d0
            max0 = c0 + d0
            c1 = cpos1[idx]
            d1 = dpos1[idx]
            min1 = c1 - d1
            max1 = c1 + d1
            if not allow_pos0_neg and lower0 < 0:
                lower0 = 0
            if max1 > (2 - chiDiscAtPi) * pi:
                max1 = (2 - chiDiscAtPi) * pi
            if min1 < (-chiDiscAtPi) * pi:
                min1 = (-chiDiscAtPi) * pi
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            cpos1_upper[idx] = max1
            cpos1_lower[idx] = min1
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0
            if max1 > pos1_max:
                pos1_max = max1
            if min1 < pos1_min:
                pos1_min = min1

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max

    if (pos1Range is not None) and (len(pos1Range) > 1):
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_maxin = pos1_max

    if (not allow_pos0_neg) and pos0_min < 0:
        pos0_min = 0

    pos0_max = pos0_maxin * EPS32
    pos1_max = pos1_maxin * EPS32

    delta0 = (pos0_max - pos0_min) / (< float > (bins0))
    delta1 = (pos1_max - pos1_min) / (< float > (bins1))

    edges0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_maxin - 0.5 * delta0, bins0)
    edges1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_maxin - 0.5 * delta1, bins1)

    with nogil:
        for idx in range(size):
            if (check_mask) and cmask[idx]:
                continue

            data = cdata[idx]
            if not isfinite(data):
                continue
            if (check_dummy) and (fabs(data - cdummy) <= ddummy):
                continue

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]
            min1 = cpos1[idx] - dpos1[idx]
            max1 = cpos1[idx] + dpos1[idx]

            if (max0 < pos0_min) or (max1 < pos1_min) or (min0 > pos0_maxin) or (min1 > pos1_maxin):
                continue

            if min0 < pos0_min:
                min0 = pos0_min
            if min1 < pos1_min:
                min1 = pos1_min
            if max0 > pos0_maxin:
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                max1 = pos1_maxin

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = <ssize_t> fbin0_min
            bin0_max = <ssize_t> fbin0_max
            bin1_min = <ssize_t> fbin1_min
            bin1_max = <ssize_t> fbin1_max

            if bin0_min == bin0_max:
                if bin1_min == bin1_max:
                    # All pixel is within a single bin
                    out_count[bin0_min, bin1_min] += 1.0
                    out_data[bin0_min, bin1_min] += data
                else:
                    # spread on more than 2 bins
                    deltaD = (< float > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (bin1_max)
                    deltaA = 1.0 / (fbin1_max - fbin1_min)

                    out_count[bin0_min, bin1_min] += deltaA * deltaD
                    out_data[bin0_min, bin1_min] += data * deltaA * deltaD

                    out_count[bin0_min, bin1_max] += deltaA * deltaU
                    out_data[bin0_min, bin1_max] += data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                        out_count[bin0_min, j] += deltaA
                        out_data[bin0_min, j] += data * deltaA

            else:
                # spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall on 1 bins in dim 1
                    deltaA = 1.0 / (fbin0_max - fbin0_min)
                    deltaL = (<float> (bin0_min + 1)) - fbin0_min
                    out_count[bin0_min, bin1_min] += deltaA * deltaL
                    out_data[bin0_min, bin1_min] += data * deltaA * deltaL
                    deltaR = fbin0_max - (<float> bin0_max)
                    out_count[bin0_max, bin1_min] += deltaA * deltaR
                    out_data[bin0_max, bin1_min] += data * deltaA * deltaR
                    for i in range(bin0_min + 1, bin0_max):
                            out_count[i, bin1_min] += deltaA
                            out_data[i, bin1_min] += data * deltaA
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    deltaL = (<float> (bin0_min + 1)) - fbin0_min
                    deltaR = fbin0_max - (<float> bin0_max)
                    deltaD = (<float> (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (<float> bin1_max)
                    deltaA = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                    out_count[bin0_min, bin1_min] += deltaA * deltaL * deltaD
                    out_data[bin0_min, bin1_min] += data * deltaA * deltaL * deltaD

                    out_count[bin0_min, bin1_max] += deltaA * deltaL * deltaU
                    out_data[bin0_min, bin1_max] += data * deltaA * deltaL * deltaU

                    out_count[bin0_max, bin1_min] += deltaA * deltaR * deltaD
                    out_data[bin0_max, bin1_min] += data * deltaA * deltaR * deltaD

                    out_count[bin0_max, bin1_max] += deltaA * deltaR * deltaU
                    out_data[bin0_max, bin1_max] += data * deltaA * deltaR * deltaU
                    for i in range(bin0_min + 1, bin0_max):
                            out_count[i, bin1_min] += deltaA * deltaD
                            out_data[i, bin1_min] += data * deltaA * deltaD
                            for j in range(bin1_min + 1, bin1_max):
                                out_count[i, j] += deltaA
                                out_data[i, j] += data * deltaA
                            out_count[i, bin1_max] += deltaA * deltaU
                            out_data[i, bin1_max] += data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                            out_count[bin0_min, j] += deltaA * deltaL
                            out_data[bin0_min, j] += data * deltaA * deltaL

                            out_count[bin0_max, j] += deltaA * deltaR
                            out_data[bin0_max, j] += data * deltaA * deltaR

        for i in range(bins0):
            for j in range(bins1):
                if out_count[i, j] > epsilon:
                    out_merge[i, j] = <float> (out_data[i, j] / out_count[i, j] / normalization_factor)
                else:
                    out_merge[i, j] = cdummy
    return numpy.asarray(out_merge).T, edges0, edges1, numpy.asarray(out_data).T, numpy.asarray(out_count).T

