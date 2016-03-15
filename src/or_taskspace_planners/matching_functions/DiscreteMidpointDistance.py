#!/usr/bin/env python

# Copyright (c) 2015, Carnegie Mellon University
# All rights reserved.
# Authors: David Butterworth <dbworth@cmu.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy


class DiscreteMidpointDistance(object):
    """
    A class for calculating the separation between two discrete paths
    at the mid-point.
    """

    def __init__(self, path1, path2, d):
        """
        Load the two paths.

        @param numpy.array path1 A path, being a 2D array
                                 (an array of point vectors),
                                 or a 3D array
                                 (an array of 2D pose arrays).
        @param numpy.array path2 Same as path1.
        @param function    d     Function defining some distance metric.
        """

        self.P = path1
        self.Q = path2

        path1_num_rows = len(path1) # works for 2D and 3D arrays
        path2_num_rows = len(path2)

        # Make sure each path has an odd number of discrete sample
        # points so we can find the mid-point
        if (path1_num_rows % 2 == 0):
            raise ValueError("path1 must have an odd number of rows, "
                             "when initializing "
                             "DiscreteMidpointDistance class." )
            return
        if (path2_num_rows % 2 == 0):
            raise ValueError("path2 must have an odd number of rows, "
                             "when initializing "
                             "DiscreteMidpointDistance class." )
            return

        self.path1_num_rows = path1_num_rows
        self.path2_num_rows = path2_num_rows

        self.d = d

        self.cm_seq = None

        self.path1_midpoint_idx = int(numpy.floor(path1_num_rows/2.0))
        self.path2_midpoint_idx = int(numpy.floor(path2_num_rows/2.0))


    def getCouplingSequence(self):
        """
        Return the coupling sequence, which is simply a one-to-one
        mapping of the indices of each path.
        """

        if self.cm_seq == None:
            # The sequence of pairs of waypoint indices
            # rows: first pair, mid-point pair, last pair
            # cols: path1 point, path2 point, distance
            # the 3rd column is used elsewhere to store the distances
            cm_seq = numpy.zeros([3, 3])
        else:
            return self.cm_seq

        # Add the indices for the first pair of points points,
        # the mid-points, and the last points
        cm_seq[0,0:2] = [0, 0]
        cm_seq[1,0:2] = [self.path1_midpoint_idx, self.path2_midpoint_idx]
        cm_seq[2,0:2] = [self.path1_num_rows-1, self.path2_num_rows-1]

        self.cm_seq = cm_seq

        # Return first two columns, which are indices
        return self.cm_seq[:,0:2].astype(int)


    def getCouplingSequenceWithDistances(self):
        """
        Compute the distance between each pair of points.
        """

        if self.cm_seq == None:
            self.getCouplingSequence()

        for i in xrange(0, len(self.cm_seq)):
            # Get the next two points in the coupling sequence cm_seq
            u = self.P[(int(self.cm_seq[i,0]))]
            v = self.Q[(int(self.cm_seq[i,1]))]

            dist = self.d(u, v)
            self.cm_seq[i,2] = dist

        return self.cm_seq


    def findMaxDeviation(self):
        """
        Get the pair of points and the separation distance at the
        mid-point.
        """

        path1_idx = self.path1_midpoint_idx
        path2_idx = self.path2_midpoint_idx

        u = self.P[(int(path1_idx))]
        v = self.Q[(int(path2_idx))]

        dist = self.d(u, v)

        return numpy.array([path1_idx,
                            path2_idx,
                            dist])
