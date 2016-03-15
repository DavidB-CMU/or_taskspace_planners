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


class DiscreteOneToOneDistance(object):
    """
    A class for calculating the separation between two discrete paths,
    using one-to-one matching with some distance metric.
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

        if path1_num_rows != path2_num_rows:
            raise ValueError("path1 and path2 arrays must have the same"
                             " number of rows, when initializing"
                             " DiscreteOneToOneDistance class." )
            return

        self.path1_num_rows = path1_num_rows
        self.path2_num_rows = path2_num_rows

        self.d = d

        self.cm_seq = None

        # The row index of cm_seq with the maximum distance
        self.max_dist_idx = None


    def getCouplingSequence(self):
        """
        Return the coupling sequence, which is simply a one-to-one
        mapping of the indices of each path.
        """

        num_rows = self.path1_num_rows

        if self.cm_seq == None:
            # The sequence of pairs of waypoint indices,
            # the 3rd column is used elsewhere to store the distances
            cm_seq = numpy.zeros([num_rows, 3])
        else:
            return self.cm_seq

        for i in xrange(0, num_rows):
            cm_seq[i,0:2] = [i, i]

        self.cm_seq = cm_seq

        # Return first two columns, which are indices
        return self.cm_seq[:,0:2].astype(int)


    def getCouplingSequenceWithDistances(self):
        """
        Compute the distance between each pair of points.
        """

        if self.cm_seq == None:
            self.getCouplingSequence()

        temp = []
        max_dist = 0.0
        max_dist_i = 0
        for i in xrange(0, len(self.cm_seq)):
            # Get the next two points in the coupling sequence cm_seq
            u = self.P[(int(self.cm_seq[i,0]))]
            v = self.Q[(int(self.cm_seq[i,1]))]

            dist = self.d(u, v)
            self.cm_seq[i,2] = dist

            # Store the row index with the max distance
            if (dist > max_dist):
                max_dist = dist
                max_dist_i = i

        self.max_dist_idx = max_dist_i

        return self.cm_seq


    def findMaxDeviation(self):
        """
        Get the pair of points and the separation distance, at the
        position of max deviation.

        Note: This returns the first pair of points only, so there
              may be other pairs of points with the same separation.
        """
        if self.max_dist_idx == None:
            self.getCouplingSequenceWithDistances()

        return self.cm_seq[int(self.max_dist_idx),:]
