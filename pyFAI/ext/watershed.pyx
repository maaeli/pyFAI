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

#
"""Inverse watershed for connecting region of high intensity
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "19/05/2017"
__status__ = "stable"
__license__ = "MIT"

import cython
import numpy
cimport numpy
import sys
import logging
from scipy.ndimage import filters 
from libc.math cimport atan2, sqrt, M_PI as pi
logger = logging.getLogger("pyFAI.ext.watershed")
from ..decorators import timeit


include "numpy_common.pxi"
include "bilinear.pxi"


cdef bint get_bit(int byteval, int idx) nogil:
    return ((byteval & (1 << idx)) != 0)


cdef class BorderPixel:
    cdef:
        readonly int index
        readonly float height
        readonly set neighbors
    
    def __cinit__(self, int idx, float height):
        self.index = idx
        self.height = height
        self.neighbors = set()
    
    def __dealloc__(self):
        self.neighbors = None

    def __repr__(self):
        return "Border #%i @%s to {%s}" % \
            (self.index, self.height, ", ".join([str(i) for i in self.neighbors]))


cdef class PeakSubRegion:
    cdef:
        readonly int index
        readonly float height, orientation
    
    def __cinit__(self, int idx, float height):
        self.index = idx
        self.height = height
        # Orientation in degree from 0 to 180°
        self.orientation = -1.0 
        
    def __repr__(self):
        return "Peak #%i @%.3f" % (self.index, self.height)
    
    cpdef set_orientation(self, float value):
        self.orientation = value


cdef class PassPixel:
    cdef:
        readonly int index, current, neighbor
        readonly float height
    
    def __cinit__(self, int idx, float height, int current, int neighbor):
        self.index = idx
        self.height = height
        self.current = current
        self.neighbor = neighbor
        
    def __repr__(self):
        return "Pass #%i @%.3f from %i to %i " % (self.index, self.height, self.current, self.neighbor)
        
        
cdef class Region:
    cdef:
        readonly int index, width
        readonly float mini, maxi
        readonly dict neighbors  # key: id of neigh value: PassPixel instance 
        readonly dict peaks  #  key: index, value PeakSubRegion instance
        readonly dict border #  key: index, value BorderPixel
        readonly list pixels  #  list of pixels in the region

    def __cinit__(self, int idx, float height, image_width):
        self.index = idx
        self.width = image_width
        self.neighbors = {}  #  key: id of neigh value: PassPixel instance 
        self.border = {}     # pixel indices of the border -> BorderPixel instance
        self.peaks = {}  
        self.peaks[idx] = PeakSubRegion(idx, height)
        self.pixels = []
        self.mini = sys.maxsize
        self.maxi = height 

    def __dealloc__(self):
        """Destructor"""
        self.neighbors = None
        self.border = None
        self.peaks = None
        self.pixels = None

    def __repr__(self):
        peaks = ", ".join([str(b) for b in self.peaks.values()])
        borders = ", ".join([str(b) for b in self.border.values()])
        nb = (", ".join(["%i@%s" % (i, j) for i, j in self.neighbors.items()]))
        lst = ["Region #%s of size %s: maxi=%s, mini=%s" % 
               (self.index, self.size, self.maxi, self.mini),
               "peaks: {%s}" % peaks,
               "border: {%s}" % borders,
               "neighbors: {%s}" % nb,
               ]
        return "\n".join(lst)
   
    def get_maxi(self):
        return self.maxi

    def get_mini(self):
        return self.mini

    @property
    def highest_pass(self):
        """Calculate and return the highest_pass

        :return: PassPixel instance 
        """
        cdef list n
        if self.neighbors:
            n = list(self.neighbors.keys())
            n.sort(key=lambda i: self.neighbors[i].height, reverse=True)
            return self.neighbors[0]
        
    def get_index(self):
        return self.index

    def get_borders(self):
        return self.border

    def get_neighbors(self):
        """Retrieve the neighbors: a dict with neighbor -> Pass instance
        :return: dict{ int -> PassPixel instance  }
        """
        return self.neighbors

    @property 
    def size(self):
        return len(self.pixels)        

    def get_size(self):
        return len(self.pixels)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def merge(self, Region other):
        """merge 2 regions
        
        :param other: region
        :return: region 
        """
        cdef:
            int i
            Region region
            BorderPixel bp
            set s, l, d
        if other.maxi > self.maxi:
            region = Region(other.index, other.maxi, self.width)
        else:
            region = Region(self.index, self.maxi, self.width)
        region.mini = min(self.mini, other.mini)
        region.pixels = self.pixels + other.pixels
        region.peaks.update(self.peaks)
        region.peaks.update(other.peaks)
        region.neighbors.update(self.neighbors)
        region.neighbors.update(other.neighbors)
        for i in region.peaks:
            region.neighbors.pop(i, None)
        l = region.peaks.keys()
        for i, bp in self.border.items() + other.border.items():
            bp.neighbors.difference_update(l)
            if bp.neighbors:
                region.border[i] = bp  
        return region

    def select_neighbors(self, int peak=-1):
        """This method is in charge of selecting the best neighbor peak for 
        merging with it
        
        :param: start measuring from given peak, not from the most intense one
        :return: index of the neighbor
        """
        cdef:
            int i, j, k, l, nb, x, y, selected 
            PassPixel pass_pixel
            float h, dist dist_min, dy, dx
            PeakSubRegion psr

        psr = self.peaks.get(peak, None)
            
        if psr is None:
            x = self.index % self.width
            y = self.index // self.width
            h = self.maxi
        else:
            x = psr.index % self.width
            y = psr.index // self.width
            h = self.height
        selected = -1
        dist_min = sys.maxint
        for nb, pass_pixel in self.neighbors.items():
            dy = pass_pixel.index // self.width - y
            dx = pass_pixel.index % self.width - x
            dist = sqrt(dy**2 + dx**2 + (h - pass_pixel.height)**2)
            angle = (atan2(dy,dx)*180.0/pi+360.0) % 180
            #co = cos()

class InverseWatershed(object):
    """
    Idea:

    * label all peaks
    * define region around those peaks which raise always to this same peak
    * define the border of such region
    * search for passes between two peaks
    * define orientation of sub-region surrounding peaks 
    * merge region with high pass between them

    """
    NAME = "Inverse watershed"
    VERSION = "2.0"

    def __init__(self, data not None, thres=1.0):
        """Constructor if the InverseWatershed class
        
        
        :param data: 2d image as numpy array
        """
        assert data.ndim == 2, "data.ndim == 2"
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)

        self.height, self.width = data.shape
        self.bilinear = Bilinear(data)
        self.regions = {}
        self.labels = None 
        self.borders = None
        self.thres = thres
        self._actual_thres = 2

    def __dealloc__(self):
        """destructor"""
        self.data = None
        self.bilinear = None
        self.regions = None
        self.labels = None
        self.borders = None
        self.dict = None

    def __repr__(self):
        return "InverseWatershed on %ix%i image with %i segmented regions (%i non empty)" % (
                self.width, self.height, len(self.regions), len([i for i in self.regions.values() if i is not None]))

    def save(self, fname):
        """
        Save all regions into a HDF5 file
        
        TODO ... broken since v1.0
        """
        raise NotImplementedError("Save is not yet implemented")
        import h5py
        with h5py.File(fname) as h5:
            h5["NAME"] = self.NAME
            h5["VERSION"] = self.VERSION
            for i in ("data", "height", "width", "labels", "borders", "thres"):
                h5[i] = self.__getattribute__(i)
            r = h5.require_group("regions")

            for i in set(self.regions.values()):
                s = r.require_group(str(i.index))
                for j in ("index", "size", "pass_to", "mini", "maxi", "highest_pass", "orientation", "border", "peaks"):
                    s[j] = i.__getattribute__(j)
                
                neighbors_dtype = numpy.dtype([("idx", numpy.int32), 
                                               ("pass", numpy.float32)])
                ary = numpy.zeros(len(i.neighbors), dtype=neighbors_dtype)
                for i, (k, v) in i.neighbors.items():
                    #ary[]
                    #n[k]
                    pass
                s["neighbors"] = ary
                    
    @classmethod
    def load(cls, fname):
        """
        Load data from a HDF5 file
        TODO ... broken since v1.0
        """
        raise NotImplementedError("load is not yet implemented")
        import h5py
        with h5py.File(fname) as h5:
            assert h5["VERSION"].value == cls.VERSION, "Version of module used for HDF5"
            assert h5["NAME"].value == cls.NAME, "Name of module used for HDF5"
            self = cls(h5["data"].value, h5["thres"].value)
            for i in ("labels", "borders"):
                setattr(self, i, h5[i].value)
            for i in h5["regions"].values():
                r = Region(i["index"].value)
                r.size = i["size"].value
                r.pass_to = i["pass_to"].value
                r.mini = i["mini"].value
                r.maxi = i["maxi"].value
                r.highest_pass = i["highest_pass"].value
                r.neighbors = list(i["neighbors"].value)
                r.border = list(i["border"].value)
                r.peaks = list(i["peaks"].value)
                for j in r.peaks:
                    self.regions[j] = r
        return self

    def init(self):
        "This method calls all the subsequent different initializations for the image"
        self.labels = self.init_labels()
        self.borders = self.init_borders()
        self.init_regions()
        self.init_orientations()
#        self.merge_singleton()
#        self.merge_twins()
#        self.merge_intense(self.thres)
        logger.info("found %s regions, after merge remains %s" % (len(self.regions), len(set(self.regions.values()))))

    @timeit
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def init_labels(self):
        """Create a "label" image where the value stored for every single pixel
        has the value of the index of the nearest peak.
        
        Initialize one "region" per peak.

        :return: the image with labels 
        """ 
        cdef:
            int i, j, width = self.width, height = self.height, idx, res
            numpy.int32_t[:, ::1] labels = numpy.zeros((height, width), dtype=numpy. int32)
            float[:, ::1] data = self.data
            dict regions = self.regions
            Bilinear bilinear = self.bilinear
        for i in range(height):
            for j in range(width):
                idx = j + i * width
                res = bilinear.c_local_maxi(idx)
                labels[i, j] = res
        return numpy.asarray(labels)
    
    @timeit
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def init_borders(self):
        """For each pixel on the border, store the orientation of to the neighbor
        
        With origin top left:
        * bit 0: up+left direction
        * bit 1: up direction
        * bit 2: up+right direction
        * bit 3: right direction
        * bit 4: down+right direction
        * bit 5: down direction
        * bit 6: down+left direction
        * bit 7: left direction
        :return: border array
        """
        cdef:
            int i, j, width = self.width, height = self.height, idx, res
            numpy.int32_t[:, ::1] labels
            numpy.uint8_t[:, ::1] borders
            numpy.uint8_t neighb
        if self.labels is None:
            self.labels = self.init_labels()
        labels = self.labels  
        borders = numpy.zeros((height, width), dtype=numpy.uint8)
        for i in range(height):
            for j in range(width):
                neighb = 0
                idx = j + i * width
                res = labels[i, j]
                if (i > 0) and (j > 0) and (labels[i - 1, j - 1] != res):
                    neighb |= 1
                if (i > 0) and (labels[i - 1, j] != res):
                    neighb |= 1 << 1
                if (i > 0) and (j < (width - 1)) and (labels[i - 1, j + 1] != res):
                    neighb |= 1 << 2
                if (j < (width - 1)) and (labels[i, j + 1] != res):
                    neighb |= 1 << 3
                if (i < (height - 1)) and (j < (width - 1)) and (labels[i + 1, j + 1] != res):
                    neighb |= 1 << 4
                if (i < (height - 1)) and (labels[i + 1, j] != res):
                    neighb |= 1 << 5
                if (i < (height - 1)) and (j > 0) and (labels[i + 1, j - 1] != res):
                    neighb |= 1 << 6
                if (j > 0) and (labels[i, j - 1] != res):
                    neighb |= 1 << 7
                borders[i, j] = neighb
        return numpy.asarray(borders)

    @timeit 
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def init_regions(self):
        """Populate the dictionary with all regions
        
        Populate each region with:
            * The set of pixels falling into it
            * The pixel indexes of the border 
            * The pass pixel position to the neighbors
            * The minimum value of the region which has to lie on the border
        """
        cdef:
            int i, j, idx, res, nb
            int height = self.height, width = self.width
            numpy.int32_t[:, ::1] labels
            numpy.uint8_t[:, ::1] borders
            float[:, ::1] data = self.data
            numpy.uint8_t neighb = 0
            Region region
            dict regions = self.regions
            float value 
            BorderPixel bp
            PassPixel pass_pixel

        if self.labels is None:
            self.labels = self.init_labels()
        labels = self.labels
        if self.borders is None:
            self.borders = self.init_borders()
        borders = self.borders
        
        for i in range(height):
            for j in range(width):
                idx = j + i * width
                neighb = borders[i, j]
                res = labels[i, j]
                if res not in regions:
                    region = Region(res, data[res // width, res % width], width)
                    regions[res] = region
                else:
                    region = regions[res]
                region.pixels.append(idx)
                if neighb == 0:
                    continue
                value = data[i, j]
                bp = BorderPixel(idx, value)
                region.border[idx] = bp
                if get_bit(neighb, 1):
                    bp.neighbors.add(labels[i - 1, j])
                if get_bit(neighb, 3):
                    bp.neighbors.add(labels[i, j + 1])
                if get_bit(neighb, 5):
                    bp.neighbors.add(labels[i + 1, j])
                if get_bit(neighb, 7):
                    bp.neighbors.add(labels[i, j - 1])
                if get_bit(neighb, 0):
                    bp.neighbors.add(labels[i - 1, j - 1])
                if get_bit(neighb, 2):
                    bp.neighbors.add(labels[i - 1, j + 1])
                if get_bit(neighb, 4):
                    bp.neighbors.add(labels[i + 1, j + 1])
                if get_bit(neighb, 6):
                    bp.neighbors.add(labels[i + 1, j - 1])
                for nb in bp.neighbors:
                    if (nb not in region.neighbors) or (value > region.neighbors[nb].height):
                        pass_pixel = PassPixel(idx, value, res, nb)
                        region.neighbors[nb] = pass_pixel
                if (value < region.mini):
                    region.mini = value

    @timeit
    def init_orientations(self):
        """Calculate the average orientation over a all regions using sobel 
        filter. This should ideally be called before any merging or cleaning-up
        """
        cdef: 
            float[:, ::1] data, Sx, Sy, histo2d, conv
            numpy.int32_t[:, ::1] labels
            numpy.int32_t[::1] forwards, backwards  
            int i, j, width = self.width, height = self.height, idx, nreg, col 
            float sx, sy, m, o, h, d, vnext, vprev, maxi 
        if self.labels is None:
            self.init_regions()
        labels = self.labels
        data = self.data
        Sx = filters.sobel(data)
        Sy = filters.sobel(data, axis=0)
        
        nreg = len(self.regions)
        histo2d = numpy.zeros((nreg, 18), numpy.float32)
        
        forwards = numpy.array(list(self.regions.keys()), dtype=numpy.int32)
        backwards = numpy.zeros(width * height, dtype=numpy.int32)
        for i in range(nreg):
            backwards[forwards[i]] = i
        
        for i in range(self.height):
            for j in range(self.width):
                sx = Sx[i, j]
                sy = Sy[i, j]
                m = sx * sx + sy * sy  # shall I put back the sqrt ?
                o = (atan2(sx, sy) * 18.0 / pi)  # orientation in 10°
                col = ( <int> floor(36.0 + o)) % 18
                histo2d[backwards[labels[i, j]], col] += m
                
        kernel = numpy.array([1., 3., 6., 7., 6., 3., 1.], dtype=numpy.float32)
        
        # convolve to smooth out noise after histogram
        conv = filters.convolve1d(histo2d, kernel, mode="wrap")
        
        for i in range(nreg):
            idx = 0
            maxi = conv[i, 0]
            for j in range(1, 18):
                m = conv[i, j]
                if m > maxi:
                    maxi = m
                    idx = j
            vprev = conv[i, idx - 1 if idx > 0 else 17]
            vnext = conv[i, idx + 1 if idx < 17 else 0]
            d = vprev - vnext  # gradiant
            h = vnext + vprev - 2 * maxi  #hessian
            if h != 0:
                d = 0.5 * d / h
            else:
                d = 0.0
            o = 10.0 * (idx + 0.5 + d) # appoximate orientation in degrees
            self.regions[forwards[i]].peaks[forwards[i]].set_orientation(o)
    
    @timeit
    def remove_singleton(self, int threshold=1):
        """Remove regions with few pixels in it as they cannot contain a valid peak
        
        :param threshold: minimal number of pixel to be in a region
        """
        cdef:
            int idx
            Region reg
        for idx in self.regions.keys(): 
            # loop over a copy
            reg = self.regions[idx]
            if reg and (reg.size <= threshold):
                self.regions[idx] = None            

    def merge(self, centile=10, orientation=45, pass_threshold=0.5):
        """Merge peak regions 

        The best neighbor is defined as the one with the peak-peak angle closest
        to the (current) region orientation.
        There are some additional tests:
        * Orientation treshold: the best neighbor's orientation is close the current
        * The height of pass is high: (pass-mini)/(maxi-mini) >=thres
        
        
        @param centile: start merging from the most intense ones but how many ?
        @param orientation: check both regions are 
        """
        cdef:
            int i, j, k, imax, imin, key1, key2, key, next_is
            float[:] flat = self.data.ravel()
            numpy.uint8_t neighb = 0
            Region region1, region2, region
            dict regions = self.regions
            float val, maxi, mini
            bint found = True
            int width = self.width
            int cnt = 0
            PassPixel pass_pixel
            list keys
            float best_angle, orientation1
        
        keys = [region.index for region in regions.values() if region is not None]
        keys.sort(revert=True, key=lambda region: region.maxi)
        size = 1 + len(keys) * centile // 100
        for key in keys[:size]:
            region1 = regions[key]
            orientation1 = region1.peaks[key].orientation
            for key2, pass_pixel in region1.neighbor.items():
                cut_height = (pass_pixel.height - region1.mini) / (region1.maxi - region1.mini)
                #orientation1
                #cut_orientation = 180 * atan2() / pi 

    def orientation_map(self):
        """Creates an orientation map and return it
        
        :return: orientation map with orientation in degrees
        """
        cdef:
            float[:, ::1] res
            numpy.int32_t[:, ::1] labels
            int width = self.width, height = self.height, idx, i, j
            dict regions = self.regions
            Region region
            float orientation
        
        if self.labels is None:
            self.init_orientation()
        labels = self.labels
        res = numpy.zeros((self.height, self.width), dtype=numpy.float32)
        for i in range(height):
            for j in range(width):
                idx = labels[i, j]
                region = regions[idx]
                if region is not None:
                    orientation = region.peaks[idx].orientation
                else:
                    orientation = numpy.NaN
                res[i, j] = orientation
        return numpy.asarray(res)

    def peaks_from_area(self, mask, Imin=None, keep=None, bint refine=True, float dmin=0.0, **kwarg):
        """
        :param mask: mask of data points valid
        :param Imin: Minimum intensity for a peak
        :param keep: Number of  points to keep
        :param refine: refine sub-pixel position
        :param dmin: minimum distance from
        """
        cdef:
            int i, j, l, x, y, width = self.width
            numpy.uint8_t[:] mask_flat = numpy.ascontiguousarray(mask.ravel(), numpy.uint8)
            int[:] input_points = numpy.where(mask_flat)[0].astype(numpy.int32)
            numpy.int32_t[:] labels = self.labels.ravel()
            dict regions = self.regions
            Region region
            list output_points = [], intensities = [], argsort, tmp_lst, rej_lst
            set keep_regions = set()
            float[:] data = self.data.ravel()
            double d2, dmin2
        for i in input_points:
            l = labels[i]
            region = regions[l]
            keep_regions.add(region.index)
        for i in keep_regions:
            region = regions[i]
            for j in region.peaks:
                if mask_flat[j]:
                    intensities.append(data[j])
                    x = j % self.width
                    y = j // self.width
                    output_points.append((y, x))
        if refine:
            for i in range(len(output_points)):
                output_points[i] = self.bilinear.local_maxi(output_points[i])
        if Imin or keep:
            argsort = sorted(range(len(intensities)), key=intensities.__getitem__, reverse=True)
            if Imin:
                argsort = [i for i in argsort if intensities[i] >= Imin]
            output_points = [output_points[i] for i in argsort]

            if dmin:
                dmin2 = dmin * dmin
            else:
                dmin2 = 0.0
            if keep and len(output_points) > keep:
                tmp_lst = output_points
                rej_lst = []
                output_points = []
                for pt in tmp_lst:
                    for pt2 in output_points:
                        d2 = (pt[0] - pt2[0]) ** 2 + (pt[1] - pt2[1]) ** 2
                        if d2 <= dmin2:
                            rej_lst.append(pt)
                            break
                    else:
                        output_points.append(pt)
                        if len(output_points) >= keep:
                            return output_points
                output_points = (output_points + rej_lst)[:keep]
        return output_points
