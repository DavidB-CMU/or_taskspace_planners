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


# Compute the discrete Hausdorff distance between two paths P and Q,
# which is the max of the directed (one-way) Hausdorff distance.
# The one-way Hausdorff distance is the max of the minimum distance (max-min)
# from each point on one path to a point on the other path.
#
# Usage:
#   (Hd, dhd, dist_mat, couple_points) = ComputeDiscreteHausdorffDistance(P, Q)
#
# Input: P,Q  Two piece-wise linear paths, being arrays where the
#             rows are the waypoints, and columns are x,y,z values
#             of each waypoint.
#
# Output: Hd   The discrete Hausdorff distance, max(dhd)
#         dhd  The directed discrete Hausdorff distances (two values)
#         dist_mat  Matrix of distances between each pair of points
#         couple_points  The end-points of each dhd pair
#
#
# David Butterworth, 2015
#

import numpy


class DiscreteHausdorffDistance(object):
    """
    A class for calculating the discrete Hausdorff distance
    between two paths.
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

        self.function1 = numpy.argmin
        self.function2 = numpy.argmax

        self.d = d

        self.distance_matrix = None


    def _computeDistanceMatrix(self):
        """
        Compute the distance matrix between two discrete sets of points.
        """

        if self.distance_matrix != None:
            # Distance matrix has already been created
            return

        P = self.P
        Q = self.Q

        #d = self.d

        P_num_rows = len(P) # works for 2D and 3D arrays
        Q_num_rows = len(Q)

        num_combinations = P_num_rows*Q_num_rows

        # Create a matrix of dimension Nx2, containing all combinations of
        # index numbers for the two paths, where N = P_num_rows * Q_num_rows
        P_i = numpy.tile(numpy.arange(P_num_rows), (1,Q_num_rows)).transpose() # column vector
        Q_i = numpy.tile(numpy.arange(Q_num_rows), (P_num_rows,1)) # matrix, rows are identical
        Q_i = numpy.reshape( numpy.ravel(Q_i, order='F'), (num_combinations,1) )
        path_idx_combinations = numpy.concatenate((P_i,Q_i), axis=1) # combine 2 column vectors horizontally

        # Get the actual point values for all combinations
        P_points_combinations = P[ path_idx_combinations[:,0], :]
        Q_points_combinations = Q[ path_idx_combinations[:,1], :]

        # Calculate the distances between all combinations of pairs of points
        distance_combinations = numpy.zeros(num_combinations)
        for i in xrange(num_combinations):
            u = P_points_combinations[i,:]
            v = Q_points_combinations[i,:]
            distance_combinations[i] = self.d(u, v)

        # Create a matrix of distances,
        # where the rows index path P
        # and the columns index path Q.
        distances = numpy.reshape(distance_combinations, (P_num_rows,Q_num_rows))

        # Store the distance matrix
        self.distance_matrix = distances


    def getHausdorffDistance(self):
        """
        Get the Hausdorff distance between paths P and Q.
        """

        self._computeDistanceMatrix()

        function1 = self.function1
        function2 = self.function2
        distances = self.distance_matrix

        # Get column vector of Function1 values of each row in the matrix,
        # where Function1 could be argmin or argmax
        P_indices = function1(distances, axis=1)
        # for each row, return the column value matching P_indices
        P_min_dist_vec = distances[numpy.arange(distances.shape[0]), P_indices]
     
        # Get row vector of Function1 values of each column in the matrix,
        # where Function1 could be argmin or argmax
        Q_indices = function1(distances, axis=0)
        # for each column, return the row value matching Q_indices
        Q_min_dist_vec = distances[Q_indices, numpy.arange(distances.shape[1])]

        # Get the Function2 value,
        # where Function2 could be armin or argmax  
        P_HD_pt_idx = function2(P_min_dist_vec)
        P_HDist = P_min_dist_vec[P_HD_pt_idx]
       
        Q_HD_pt_idx = function2(Q_min_dist_vec)
        Q_HDist = Q_min_dist_vec[Q_HD_pt_idx]

        # The forward and backward Hausdorff distances (two values)
        dhd = numpy.array([P_HDist, Q_HDist])

        # The Hausdorff distance
        Hd = numpy.amax(dhd)
        Hd_idx = function2(dhd)

        return Hd


    def getDirectedHausdorffDistances(self):
        """
        Get the forward and backward Hausdorff distances between
        paths P and Q.
        """

        self._computeDistanceMatrix()

        function1 = self.function1
        function2 = self.function2
        distances = self.distance_matrix

        # Get column vector of Function1 values of each row in the matrix,
        # where Function1 could be argmin or argmax
        P_indices = function1(distances, axis=1)
        # for each row, return the column value matching P_indices
        P_min_dist_vec = distances[numpy.arange(distances.shape[0]), P_indices]
     
        # Get row vector of Function1 values of each column in the matrix,
        # where Function1 could be argmin or argmax
        Q_indices = function1(distances, axis=0)
        # for each column, return the row value matching Q_indices
        Q_min_dist_vec = distances[Q_indices, numpy.arange(distances.shape[1])]

        # Get the Function2 value,
        # where Function2 could be armin or argmax  
        P_HD_pt_idx = function2(P_min_dist_vec)
        P_HDist = P_min_dist_vec[P_HD_pt_idx]
       
        Q_HD_pt_idx = function2(Q_min_dist_vec)
        Q_HDist = Q_min_dist_vec[Q_HD_pt_idx]

        # The forward and backward Hausdorff distances (two values)
        dhd = numpy.array([P_HDist, Q_HDist])

        return dhd


    def findMaxDeviation(self):
        """
        Get the directed (forward) Hausdorff distance from path P to Q.
        """

        self._computeDistanceMatrix()

        function1 = self.function1
        function2 = self.function2
        distances = self.distance_matrix

        # Get column vector of Function1 values of each row in the matrix,
        # where Function1 could be argmin or argmax
        P_indices = function1(distances, axis=1)
        # for each row, return the column value matching P_indices
        P_min_dist_vec = distances[numpy.arange(distances.shape[0]), P_indices]
     
        # Get row vector of Function1 values of each column in the matrix,
        # where Function1 could be argmin or argmax
        Q_indices = function1(distances, axis=0)
        # for each column, return the row value matching Q_indices
        Q_min_dist_vec = distances[Q_indices, numpy.arange(distances.shape[1])]

        # Get the Function2 value,
        # where Function2 could be armin or argmax  
        P_HD_pt_idx = function2(P_min_dist_vec)
        P_HDist = P_min_dist_vec[P_HD_pt_idx]

        return numpy.array([P_indices[P_HD_pt_idx], P_HD_pt_idx, P_HDist])


    def getForwardCouplingSequence(self):
        """
        Get the forward coupling sequence from Path P.
        """
        Hd_mat = self.distance_matrix
        P = self.P
        Q = self.Q

        Hd_P_indices = numpy.argmin(Hd_mat, axis=1) # a row vector

        P_num_rows = len(P)
        P_indices = numpy.arange(P_num_rows)

        Q_indices = Hd_P_indices

        # Create Nx2 array, where first column is indices of path P,
        # and second column is indices of path Q.
        return numpy.transpose( numpy.vstack((Q_indices,P_indices)) )


    def getCouplingSequence(self):
        """
        """
        pass
