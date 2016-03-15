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


# Calculate the discrete Frechet distance (the coupling measure)
# between two paths P and Q.
#
# Usage:
#   dfd = DiscreteFrechetDistance(P, Q)
#   cm = dfd.getCouplingMeasure()
#   cm_seq = dfd.getCouplingSequence()
#
# Input: Two piece-wise linear paths P and Q
#        Arrays where the rows are the waypoints, and columns are 
#        x,y,z values of each waypoint.
#
# Output: cm The coupling measure
#         cm_seq The coupling sequence
#
#
# David Butterworth, 2015
#
# Algorithm by:
# T. Eiter and H. Mannila, "Computing Discrete Frechet Distance",
# Technical Report CD-TR 94/64, Christian Doppler Laboratory for Expert
# Systems, Vienna University of Technology, 1994
#
# Back-tracking algorithm by:
# Zachary Danziger, 2011
#


import numpy


class DiscreteFrechetDistance(object):
    """
    A class for calculating the discrete Frechet distance
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

        path1_rows = len(path1) # works for 2D and 3D arrays
        path2_rows = len(path2)
        self.p = path1_rows # Number of points in P
        self.q = path2_rows # Number of points in Q

        self.d = d

        self.cm = None
        self.cm_seq = None

        # ca: array, this gets initialized below
        self.ca = None


    def getCouplingMeasure(self):
        """
        Return the coupling measure (the discrete Frechet distance).       
        """

        def c(i,j):
            """
            function c(i,j)
            Calculate the coupling measure between two paths P and Q.
            by iterating over the waypoints i = 0 to (p-1) on path P, 
            and j = 0 to (q-1) on path Q.

            Note: The resulting coupling sequence numbers the first
                  waypoint as 0, whereas the original published
                  algorithm starts from 1.
            """

            if (self.ca[i,j] > -1):
                return self.ca[i,j]
            elif (i == 0) and (j == 0):
                self.ca[i,j] = self.d(self.P[0],self.Q[0])
                return self.ca[i,j]
            elif (i > 0) and (j == 0):
                self.ca[i,j] = max( c(i-1,0), self.d(self.P[i],self.Q[0]) )
                return self.ca[i,j]
            elif (i == 0) and (j > 0):
                self.ca[i,j] = max( c(0,j-1), self.d(self.P[0],self.Q[j]) )
                return self.ca[i,j]
            elif (i > 0) and (j > 0):
                self.ca[i,j] = max( min([c(i-1,j), c(i-1,j-1), c(i,j-1)]), self.d(self.P[i],self.Q[j]) )
                return self.ca[i,j]
            else:
                self.ca[i,j] = numpy.inf

        if self.ca == None:
            # Initialize array ca: [1..p, 1..q]
            # a square matrix of size p x q
            # (all values must be -1 at start)
            self.ca = -1 * numpy.ones([self.p, self.q]);

        if self.cm == None:
            # Get the coupling measure
            self.cm = c((self.p - 1), (self.q - 1))

        return self.cm



    # ToDO: make it so you can't call this function without initializing the class first
    #
    # because it uses  self.ca
    # Check if other functions use self.ca ??


    # TODO:
    # Add function   getCouplingSequence(calculate_distances=True, remove_endpoints=True)



    def getCouplingSequence(self):
        """
        Return the coupling sequence, which is the sequence of steps
        along each path that result in the coupling measure.
        """

        if self.ca == None:
            self.getCouplingMeasure()

        if self.cm_seq == None:
            # The sequence of pairs of waypoint indices,
            # the 3rd column is used elsewhere to store the distances
            cm_seq = numpy.zeros([self.p+self.q+1, 3])
        else:
            return self.cm_seq

        # Create matrix where first row and first column are inf,
        # with matrix ca contained within.
        padded_ca = numpy.inf * numpy.ones([self.p+1, self.q+1])
        padded_ca[1:(self.p+1), 1:(self.q+1)] = self.ca

        Pi = self.p
        Qi = self.q
        count = 0
    
        # Iterate from p down to 2, and q down to 2
        while ((Pi != 1) or (Qi != 1)):

            # Step down the gradient of matrix padded_ca
            min_idx = numpy.argmin([padded_ca[Pi-1,Qi], padded_ca[Pi-1,Qi-1], padded_ca[Pi,Qi-1]])

            if (min_idx == 0):
                #cm_seq[count,0:2] = [Pi-1, Qi]
                cm_seq[count,0:2] = [Pi-2, Qi-1]
                Pi = Pi - 1
            elif (min_idx == 1):
                #cm_seq[count,0:2] = [Pi-1, Qi-1]
                cm_seq[count,0:2] = [Pi-2, Qi-2]
                Pi = Pi - 1
                Qi = Qi - 1
            elif (min_idx == 2):
                #cm_seq[count,0:2] = [Pi, Qi-1]
                cm_seq[count,0:2] = [Pi-1, Qi-2]
                Qi = Qi - 1

            count = count + 1

        # Find the index of the last non-zero value in cm_seq
        last_value_idx = 0
        for i in xrange(0, len(cm_seq)):
            if (cm_seq[i,0] == 0):
                last_value_idx = i - 1
                break

        # Get the non-zero values
        cm_seq = cm_seq[0:(last_value_idx+1),:]

        # Flip order of rows from bottom-to-top
        cm_seq = cm_seq[::-1]

        # Add the first and last points of P and Q
        # as pairs in the sequence
        cm_seq = numpy.vstack(([0, 0, 0.0], cm_seq))
        cm_seq = numpy.vstack((cm_seq, [self.p-1, self.q-1, 0.0]))

        self.cm_seq = cm_seq

        # Return first two columns, which are indices
        return self.cm_seq[:,0:2].astype(int)


    def getCouplingSequenceWithDistances(self):
        """
        Compute the distance between each matched pair of points in
        the Frechet coupling sequence.
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


    def getCouplingPosition(self):
        """
        Get the pair of points for which the separation distance
        is the discrete Frechet distance.

        Note: This returns the first pair of points only, so there
              may be other pairs of points with the same separation.
        """

        if self.cm_seq == None:
            self.getCouplingSequence()

        c_point = numpy.array([])
        for i in xrange(0, len(self.cm_seq)):
            # Get the next two points in the coupling sequence cm_seq
            u = self.P[(int(self.cm_seq[i,0]))]
            v = self.Q[(int(self.cm_seq[i,1]))]

            dist = self.d(u, v)

            #if dist >= self.cm:
            if numpy.isclose(dist, self.cm, rtol=1e-05, atol=1e-08, equal_nan=False):
                c_point = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
                break

        return c_point


    def getAllCouplingPositions(self):
        """
        Get all pairs of points for which the separation distance
        is the discrete Frechet distance.
        """
        if self.cm_seq == None:
            self.getCouplingSequence()

        temp = []
        for i in xrange(0, len(self.cm_seq)):
            # Get the next two points in the coupling sequence cm_seq
            u = self.P[(int(self.cm_seq[i,0]))]
            v = self.Q[(int(self.cm_seq[i,1]))]

            dist = self.d(u, v)
            print "dist = ", dist

            #if dist >= self.cm:
            if numpy.isclose(dist, self.cm, rtol=1e-05, atol=1e-08, equal_nan=False):

                c_point = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
                temp.append(c_point)

        # Return points as a vertically-stacked array
        return numpy.array(temp)


    def findMaxDeviation(self):
        """
        Get the pair of points and the separation distance, at the
        position of max deviation, based on the Frechet metric.

        Note: This returns the first pair of points only, so there
              may be other pairs of points with the same separation.
        """
        return self.getCouplingPosition()



"""
    def getFrechetPositions(self):

        getMaxFrechetIndex
        as above, but don't include 1st and last points
        get the position with the max frechet dist
        test to make sure self.cm_seq exists first.
        *now works with transforms


        #print '\n in getMaxFrechetPosition() '
        #print ' self.cm_seq = ', self.cm_seq

        # Determine the first and last point of paths P and Q
        P_first_point_idx = 0
        P_last_point_idx  = self.cm_seq[-1,0]
        Q_first_point_idx = 0
        Q_last_point_idx  = self.cm_seq[-1,1]
        #print '    P is %i:%i   Q is %i:%i' % (P_first_point_idx, P_last_point_idx, Q_first_point_idx, Q_last_point_idx)

        # Filter the coupling sequence to remove
        # the first and last point of either path,
        # which could appear more than once
        temp = []
        max_dist = 0.0
        max_dist_i = 0
        new_row_i = 0 # index of number of rows added
        for i in range(0, len(self.cm_seq)):
            #print 'i = ', i

            # Ignore the start and end points of P and Q
            #if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
            #    continue
            #elif (self.cm_seq[i,1] == Q_first_point_idx) or (self.cm_seq[i,1] == Q_last_point_idx):
            #    continue
            # only ignore on P
            if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
                continue


            #print 'after continue'

            # Calculate discrete Frechet distance for this pair of points
            P_idx = int(self.cm_seq[i,0])
            u = self.P[(P_idx)]
            Q_idx = int(self.cm_seq[i,1])
            v = self.Q[(Q_idx)]
            #dist = numpy.sqrt(numpy.sum((u-v)**2))
            dist = self.d(u, v)
            #print 'dist = ', dist

            # Store the row index with the max distance
            if (dist > max_dist):
                max_dist = dist
                max_dist_i = new_row_i 

            # Add row with point numbers ...
            #new_row = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
            new_row = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist]) # use indexes
            temp.append(new_row)
            new_row_i = new_row_i + 1

        #print '\n in getMaxFrechetPosition() len(temp) = ', len(temp), ' \n'
        print "temp = ", temp

        # Make sure we have some points
        if len(temp) == 0:
            raise ValueError('After filtering the coupling sequence to remove '
                             'the start/end points of P and Q, there are no '
                             'couples left! Try adding more samples.')

        # Convert list back to an array
        filtered_cm_seq = numpy.array(temp)

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'max_dist_i = ', max_dist_i

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'Max Frechet distance = ', max_dist, ' at row index ', max_dist_i

        return filtered_cm_seq[max_dist_i,:]







    def getMaxFrechetPosition1(self, d):

        get the position with the max frechet dist
        test to make sure self.cm_seq exists first.
        Should return aray of Ints
        *Now works with Transforms

        #print '\n in getMaxFrechetPosition() '
        #print ' self.cm_seq = ', self.cm_seq

        # Determine the first and last point of paths P and Q
        P_first_point_idx = 1
        P_last_point_idx  = self.cm_seq[-1,0]
        Q_first_point_idx = 1
        Q_last_point_idx  = self.cm_seq[-1,1]
        #print '    P is %i:%i   Q is %i:%i' % (P_first_point_idx, P_last_point_idx, Q_first_point_idx, Q_last_point_idx)

        # dont filter any points
        temp = []
        max_dist = 0.0
        max_dist_i = 0
        new_row_i = 0 # index of number of rows added
        for i in range (0, len(self.cm_seq)):
            #print 'i = ', i

            # Ignore the start and end points of P and Q
            #if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
            #    continue
            #elif (self.cm_seq[i,1] == Q_first_point_idx) or (self.cm_seq[i,1] == Q_last_point_idx):
            #    continue

            #print 'after continue'

            # Calculate discrete Frechet distance for this pair of points
            P_idx = int(self.cm_seq[i,0]) - 1 # minus one because P is indexed from zero
            u = self.P[(P_idx)]
            Q_idx = int(self.cm_seq[i,1]) - 1 # minus one because Q is indexed from zero
            v = self.Q[(Q_idx)]
            #dist = numpy.sqrt(numpy.sum((u-v)**2))
            dist = d(u, v)
            #print 'dist = ', dist

            # Store the row index with the max distance
            if (dist > max_dist):
                max_dist = dist
                max_dist_i = new_row_i 

            # Add row with point numbers ...
            new_row = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
            temp.append(new_row)
            new_row_i = new_row_i + 1

        #print '\n in getMaxFrechetPosition1() len(temp) = ', len(temp), ' \n'

        # Make sure we have some points
        if len(temp) == 0:
            raise ValueError('After ... '
                             'the start/end points of P and Q, there are no '
                             'couples left! Try adding more samples.')

        # Convert list back to an array
        filtered_cm_seq = numpy.array(temp)

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'max_dist_i = ', max_dist_i

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'Max Frechet distance = ', max_dist, ' at row index ', max_dist_i

        return filtered_cm_seq[max_dist_i,:]





    def getMaxFrechetPosition(self, d):

        getMaxFrechetIndex
        as above, but don't include 1st and last points
        get the position with the max frechet dist
        test to make sure self.cm_seq exists first.
        *now works with transforms

        #print '\n in getMaxFrechetPosition() '
        #print ' self.cm_seq = ', self.cm_seq

        # Determine the first and last point of paths P and Q
        P_first_point_idx = 1
        P_last_point_idx  = self.cm_seq[-1,0]
        Q_first_point_idx = 1
        Q_last_point_idx  = self.cm_seq[-1,1]
        #print '    P is %i:%i   Q is %i:%i' % (P_first_point_idx, P_last_point_idx, Q_first_point_idx, Q_last_point_idx)

        # Filter the coupling sequence to remove
        # the first and last point of either path,
        # which could appear more than once
        temp = []
        max_dist = 0.0
        max_dist_i = 0
        new_row_i = 0 # index of number of rows added
        for i in range(0, len(self.cm_seq)):
            #print 'i = ', i

            # Ignore the start and end points of P and Q
            #if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
            #    continue
            #elif (self.cm_seq[i,1] == Q_first_point_idx) or (self.cm_seq[i,1] == Q_last_point_idx):
            #    continue
            # only ignore on P
            if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
                continue


            #print 'after continue'

            # Calculate discrete Frechet distance for this pair of points
            P_idx = int(self.cm_seq[i,0]) - 1 # minus one because P is indexed from zero
            u = self.P[(P_idx)]
            Q_idx = int(self.cm_seq[i,1]) - 1 # minus one because Q is indexed from zero
            v = self.Q[(Q_idx)]
            #dist = numpy.sqrt(numpy.sum((u-v)**2))
            dist = d(u, v)
            #print 'dist = ', dist

            # Store the row index with the max distance
            if (dist > max_dist):
                max_dist = dist
                max_dist_i = new_row_i 

            # Add row with point numbers ...
            #new_row = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
            new_row = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist]) # use indexes
            temp.append(new_row)
            new_row_i = new_row_i + 1

        #print '\n in getMaxFrechetPosition() len(temp) = ', len(temp), ' \n'

        # Make sure we have some points
        if len(temp) == 0:
            raise ValueError('After filtering the coupling sequence to remove '
                             'the start/end points of P and Q, there are no '
                             'couples left! Try adding more samples.')

        # Convert list back to an array
        filtered_cm_seq = numpy.array(temp)

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'max_dist_i = ', max_dist_i

        #print 'filtered_cm_seq = ', filtered_cm_seq
        #print 'Max Frechet distance = ', max_dist, ' at row index ', max_dist_i

        return filtered_cm_seq[max_dist_i,:]
"""



"""
# This was with function above

c_point = numpy.array([])
for i in range (0, len(self.cm_seq)):
    print 'getCouplingPosition2, %i = ' % i
    u = self.P[(self.cm_seq[i,0]-1),:] # minus one because P is indexed from zero
    v = self.Q[(self.cm_seq[i,1]-1),:] # minus one because Q is indexed from zero

    #print 'P_i = %i' % i
    # Ignore start and end points of P
    if (self.cm_seq[i,0] == P_first_point_idx) or (self.cm_seq[i,0] == P_last_point_idx):
        print 'continue 1'
        continue
    elif (self.cm_seq[i,1] == Q_first_point_idx) or (self.cm_seq[i,1] == Q_last_point_idx):
        print 'continue 2'
        continue

    
    dist = numpy.sqrt(numpy.sum((u-v)**2))
    print 'dist = ', dist
    print 'self.cm = ', self.cm
    #if dist >= self.cm:
    if dist >= 0.0:
        c_point = numpy.array([self.cm_seq[i,0], self.cm_seq[i,1], dist])
        print 'c_point = ', c_point
        break

return c_point
"""




