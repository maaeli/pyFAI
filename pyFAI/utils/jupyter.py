# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""Jupyter helper function
"""

from __future__ import division, print_function

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/02/2017"
__status__ = "Development"
__docformat__ = 'restructuredtext'

import numpy
from pylab import figure, legend


def display_cp(img, cp=None, ai=None):
    """Display an image with the control points and the calibrated rings
    in jupyter
    
    :param img: 2D numpy array with an image
    :param cp: ControlPoint instance
    :param ai: azimuthal integrator for iso-2th curves 
    """
    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(numpy.arcsinh(img), origin="lower")
    if cp is not None:
        for lbl in cp.get_labels():
            pt = numpy.array(cp.get(lbl=lbl).points)
            ax.scatter(pt[:, 1], pt[:, 0], label=lbl)
        if ai is not None:
            tth = cp.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(ttha, levels=tth, cmap="autumn", linewidths=2, linestyles="dashed")
        legend()
    return fig