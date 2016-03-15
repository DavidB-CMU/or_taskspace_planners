#!/usr/bin/env python

import logging
import numpy
import openravepy
import time
from prpy.util import SetTrajectoryTags

from prpy.planning.base import (BasePlanner,
                                PlanningError,
                                PlanningMethod,
                                Tags)

import prpy.util

from prpy.planning.exceptions import (CollisionPlanningError,
                                      SelfCollisionPlanningError)
#                                      ConstraintViolationPlanningError,
#                                      JointLimitError)

logger = logging.getLogger(__name__)

#
# Recursive Bounded Deviation using Matching Function & Distance Metric
#
# ToDo:
# russell_bounded_deviation "bounded deviation joint paths"
#

#---------------------------------------------------------------------#

# Helper functions used by the planner:


# Previously called ArrayToTraj()
def CreateJointSpaceTrajectory(robot, q_list, dof_indices=None):
    """
    Create a joint-space trajectory using linear interpolation between
    joint configurations.
    """

    if dof_indices == None:
        dof_indices = robot.GetActiveDOFIndices()

    env = robot.GetEnv()

    traj = openravepy.RaveCreateTrajectory(env, '')
    cspec = robot.GetActiveConfigurationSpecification('linear')
    traj.Init(cspec)

    for i in xrange(len(q_list)):
        waypoint = numpy.zeros(cspec.GetDOF())
        cspec.InsertJointValues(waypoint, q_list[i], robot,
                                dof_indices, False)
        traj.Insert(i, waypoint.ravel())

    return traj


def SampleWorkspaceTrajectoryIntoArray(robot, traj, t_start, t_end, n):
    """
    Takes n samples from a workspace trajectory, including the
    end-points, and returns them in an array.
    """

    if t_start < 0.0:
        raise ValueError("t_start < 0.0 "
                         "in SampleWorkspaceTrajectoryIntoArray()")
    if t_end <= t_start:
        raise ValueError("t_end <= t_start "
                         "in SampleWorkspaceTrajectoryIntoArray()")
    num_samples = n
    if num_samples <= 3:
        raise ValueError("Number of samples 'n' must be >= 3 "
                         "in SampleWorkspaceTrajectoryIntoArray()")

    cspec = traj.GetConfigurationSpecification()
    dof_indices, _ = cspec.ExtractUsedIndices(robot)

    if t_end == None:
        t_end = traj.GetDuration()

    times_list = []
    transforms_list = []
    for t in numpy.linspace(t_start, t_end, num_samples):
        times_list.append(t)
        P = traj.Sample(t)[0:7] # first 7 values are pose
        T_current = openravepy.matrixFromPose(P)
        transforms_list.append(T_current)

    times_array = numpy.asarray(times_list)
    transforms_array = numpy.asarray(transforms_list)

    return times_array,transforms_array


def CreateWorkspaceTrajectoryByLinearInterpBetweenConfigs(env, robot,
                                                          q0, q1):
    """
    Create a workspace trajectory by linearly interpolating
    between two joint configurations.
    """

    # Get a jointspace trajectory between the two configurations
    # using linear joint interpolation
    path = None
    samples = None
    with robot: # save robot state
        _,samples = InterpBetweenConfigs(env, robot, q0, q1)
    # restore robot state

    traj = openravepy.RaveCreateTrajectory(env, '')
    velocity_interp = 'linear'
    spec = openravepy.IkParameterization.\
                            GetConfigurationSpecificationFromType(
                                 openravepy.IkParameterizationType.Transform6D,
                                 velocity_interp)
    traj.Init(spec)

    # Create a workspace path of end-effector poses.
    # Each joint configuration 'q' sampled is L2 norm
    # from the previous configuration.
    for idx,(t,q) in enumerate(samples):
        T_current = prpy.util.GetForwardKinematics(robot, q)
        traj.Insert(idx, openravepy.poseFromMatrix(T_current))

    # If q0 and q1 are close together, only q0 will exist in 'samples'
    # and the workspace trajectory will only have one waypoint,
    # there we manually add q1 as a second waypoint.
    num_waypoints = traj.GetNumWaypoints()
    if num_waypoints == 1:
        T_goal = prpy.util.GetForwardKinematics(robot, q1)
        traj.Insert(num_waypoints, openravepy.poseFromMatrix(T_goal))

    # Add timing so we can continuously sample the trajectory
    openravepy.planningutils.RetimeAffineTrajectory(traj, maxvelocities=0.1*numpy.ones(7), maxaccelerations=0.1*numpy.ones(7))

    return traj


def CreateJointTrajectory(env, robot, q_start, q_goal, dof_indices=None):
    """
    Create a jointspace trajectory between two configurations.
    """

    if dof_indices == None:
        dof_indices = robot.GetActiveDOFIndices()

    traj = openravepy.RaveCreateTrajectory(env, '')
    velocity_interp = 'linear'
    cspec = robot.GetActiveConfigurationSpecification(velocity_interp)

    # Add first waypoint
    start_waypoint = numpy.zeros(cspec.GetDOF())
    cspec.InsertJointValues(start_waypoint, q_start, robot, dof_indices, False)
    traj.Init(cspec)
    traj.Insert(0, start_waypoint.ravel())

    # Make the trajectory end at the goal configuration, as
    # is not identical to the start configuration.
    if not numpy.allclose(q_start, q_goal):
        goal_waypoint = numpy.zeros(cspec.GetDOF())
        cspec.InsertJointValues(goal_waypoint, q_goal, robot,
                                dof_indices, False)
        traj.Insert(1, goal_waypoint.ravel())

    return traj


def InterpBetweenConfigs(env, robot, start, goal):
    """
    Generate joint configurations that are L2 norm apart, between a
    start and goal configuration using linear interpolation.

    This function is similar to SnapPlan() except it returns the
    actual configurations and does not do collision checking.

    @returns (openravepy.Trajectory, generator) A jointspace trajectory
                                                having the start and
                                                goal configuration,
                                                a generator of samples
                                                (t,q) being
                                                the sample time 't'
                                                joint configuration 'q'.
    """

    # Check that the start and goal are within joint limits.
    prpy.util.CheckJointLimits(robot, start)
    prpy.util.CheckJointLimits(robot, goal)

    traj = CreateJointTrajectory(env, robot, start, goal)

    # Get generator of joint configurations (t,q)
    # this may contain 1 or more samples.
    stg = prpy.util.SampleTimeGenerator
    samples = prpy.util.GetLinearCollisionCheckPts(robot, \
                                                   traj, \
                                                   norm_order=2, \
                                                   sampling_func=stg)
    return traj,samples # trajectory, tuple (t,q)


"""
def renderTrajectoryPoses(rviz_tools_class, robot, traj, num_samples=30):

    do blah

    @param instance rviz_tools_class: An instance of the rviz_tools_class.RvizMarkers class

    cspec = traj.GetConfigurationSpecification()
    dof_indices, _ = cspec.ExtractUsedIndices(robot)
    #q_resolutions = robot.GetDOFResolutions()[dof_indices]
    duration = traj.GetDuration()
    #print 'duration = ', duration

    #num_samples = 30
    for t in numpy.linspace(0, duration, num_samples):

        T_ee_curr = openravepy.matrixFromPose( traj.Sample(t)[0:7] )
        rviz_tools_class.publishAxis(T_ee_curr, 0.02, 0.001, None) # pose, axis length, radius, lifetime
"""


def GetWorkspaceTrajectoryWaypointTimes(traj):
    """
    Return an array containing the time at each waypoint
    in a workspace trajectory.
    """
    cspec = traj.GetConfigurationSpecification()
    waypoint_times = list()
    dt = 0.0
    waypoint_times.append(dt)
    for i in xrange(1, traj.GetNumWaypoints()):
        dt += cspec.ExtractDeltaTime(traj.GetWaypoint(i))
        waypoint_times.append(dt)
    return numpy.asarray(waypoint_times)


#---------------------------------------------------------------------#

# Helper functions for visualization:


def PublishWorkspaceTrajectorySegments(rviz_tools, robot, traj,
                                       color='red', line_width=0.005):
    """
    Draw the linear segment between each waypoint of a workspace
    trajectory as a line using Rviz Markers.

    This works because the trajectory has linear interpolation for the
    translation component.

    @param instance    rviz_tools: An instance of the rviz_tools.RvizMarkers class
    @param openravepy.robot robot: The robot.
    @param openravepy.Trajectory traj: A workspace trajectory.
    @param color     string: Color name to draw the line segments.
    @param line_width float: Width of the line segments.
    """

    num_waypoints = traj.GetNumWaypoints()
    T_prev = openravepy.matrixFromPose(traj.GetWaypoint(0)[0:7])
    for i in range(1, num_waypoints):
        P = traj.GetWaypoint(i)[0:7] # first 7 values are pose
        T = openravepy.matrixFromPose(P)
        duration = None # None = publish forever
        rviz_tools.publishLine(T_prev, T, color, line_width, duration)
        T_prev = T


def PublishWorkspaceTrajectoryClosestSegment(rviz_tools, robot, traj,
                                             color='red', line_width=0.005,
                                             t_min=None, t_max=None):
    """
    Draw the linear segment of a workspace trajectory which is closest
    to the segment from t_min to t_max.
    """

    if (t_min == None) or (t_max == None):
        raise ValueError("You must specify the desired segment using "
                         "t_min and t_max in "
                         "PublishWorkspaceTrajectoryClosestSegment()")

    waypoint_times = GetWorkspaceTrajectoryWaypointTimes(traj)

    epsilon = 0.000001
    
    T_prev = openravepy.matrixFromPose(traj.GetWaypoint(0)[0:7])
    waypoint_time_prev = 0.0
    num_waypoints = traj.GetNumWaypoints()
    for i in range(1, num_waypoints):
        P = traj.GetWaypoint(i)[0:7] # first 7 values are pose
        T = openravepy.matrixFromPose(P) 
        waypoint_time = waypoint_times[i]
        diff_from_t_min = abs(waypoint_time_prev - t_min)
        diff_from_t_max = abs(waypoint_time - t_max)

        # If this segment starts and ends outside the desired time
        # limits then draw it
        if (t_min+epsilon >= waypoint_time_prev) \
                                          and (t_max-epsilon <= waypoint_time):
            duration = None # None = publish forever
            rviz_tools.publishLine(T_prev, T, color, line_width, duration)

        T_prev = T
        waypoint_time_prev = waypoint_time


# Between two values of t
"""
def PublishWorkspaceTrajectorySegments(rviz_tools, robot, traj, color='red', line_width=0.005, t_min=None, t_max=None):
    print "in PublishWorkspaceTrajectorySegments()"
    print "t_min = ", t_min
    print "t_max = ", t_max

    waypoint_times = GetWorkspaceTrajectoryWaypointTimes(traj)

    epsilon = 0.000001

    num_waypoints = traj.GetNumWaypoints()
    #T_prev = openravepy.matrixFromPose(traj.Sample(0)[0:7])
    T_prev = openravepy.matrixFromPose(traj.GetWaypoint(0)[0:7])
    waypoint_time_prev = 0.0
    for i in range(1, num_waypoints):

        T = openravepy.matrixFromPose(traj.GetWaypoint(i)[0:7]) # first 7 values are pose
        #rviz_tools.publishLine(T_prev, T, color, line_width, 0.5) #None) # None = publish forever
        

        waypoint_time = waypoint_times[i]
        diff_from_t_min = abs(waypoint_time_prev - t_min)
        diff_from_t_max = abs(waypoint_time - t_max)
        print "diff_from_t_min = ", diff_from_t_min
        print "diff_from_t_max = ", diff_from_t_max
        #if (diff_from_t_min < epsilon) and (diff_from_t_max < epsilon):
        if (waypoint_time_prev+epsilon >= t_min) and (waypoint_time-epsilon <= t_max):
            print "publishing segment ", i

            #T = openravepy.matrixFromPose(traj.GetWaypoint(i)[0:7]) # first 7 values are pose
            rviz_tools.publishLine(T_prev, T, color, line_width, None) # None = publish forever
            #T_prev = T
        T_prev = T
        waypoint_time_prev = waypoint_time
"""

def PublishWorkspaceTrajectoryPoses(rviz_tools, robot, traj, n=30, axis_length=0.02, axis_radius=0.001):
    """
    Sample a workspace trajectory n number of times and publish an axes
    Rviz Marker representing the end effector pose at each sampled
    position.

    @param instance    rviz_tools: An instance of the rviz_tools.RvizMarkers class
    @param openravepy.robot robot: The robot.
    @param openravepy.Trajectory traj: A timed workspace trajectory.
    @param n int: Number of axes to draw along the path of the
                  trajectory, including end-points.
    @param axis_length float: The length of each of the 3 axis lines. 
    @param axis_radius float: The radius of each of the 3 axis lines.
    """

    if n <= 0:
        raise ValueError("Number of samples n must be a positive number"
                         " in PublishWorkspaceTrajectoryPoses()")
    if not prpy.util.IsTimedTrajectory(traj):
        raise ValueError(
            'Trajectory must be timed. If you want to use this function on a'
            ' path, then consider using util.ComputeUnitTiming to compute its'
            ' arclength parameterization.')

    # Draw specified number of axis markers along
    # the path of the trajectory
    duration = traj.GetDuration()
    for t in numpy.linspace(0, duration, n):
        P = traj.Sample(t)[0:7] # first 7 values are pose
        T_ee_curr = openravepy.matrixFromPose(P)
        duration = None # None = publish forever
        rviz_tools.publishAxis(T_ee_curr, axis_length, axis_radius, duration)


def Convert3DArrayToListOfArrays(a):
    """
    Convert an array of 2D arrays into a list of 2D arrays.
    """
    (outer_rows,rows,cols) = numpy.shape(a)
    new_list = []
    for i in xrange(0, outer_rows):
        new_list.append( a[i,:] )
    return new_list


#---------------------------------------------------------------------#


class BoundedDeviationPlanner(BasePlanner):
    def __init__(self):
        super(BoundedDeviationPlanner, self).__init__()

        self.workspace_traj = None
        self.robot = None
        self.joint_resolutions = None
        self.max_deviation_threshold = 0.0
        self.matching_fn_class = None
        self.distance_metric_fn = None
        self.ignore_traj_segments = True
        self.recursion_limit = 0
        self.terminate_on_max_recursion = True
        self.rviz_tools_class = None     

        # Statistics
        self.deepest_recursion = 0
        self.num_collision_checks = 0


    def __str__(self):
        return 'BoundedDeviationPlanner'


    def getNumCollisionChecks(self):
        return self.num_collision_checks


    # Not tested yet
    """
    @PlanningMethod
    def PlanToEndEffectorPose(self, robot, goal_pose, timelimit=5.0,
                              **kw_args):

        Plan to an end effector pose by first creating a geodesic
        trajectory in SE(3) from the starting end-effector pose to the goal
        end-effector pose, and then attempting to follow it exactly
        using PlanWorkspacePath.

        @param robot
        @param goal_pose desired end-effector pose
        @return traj


        with robot:
            # Create geodesic trajectory in SE(3)
            manip = robot.GetActiveManipulator()
            start_pose = manip.GetEndEffectorTransform()
            traj = openravepy.RaveCreateTrajectory(self.env, '')
            spec = openravepy.IkParameterization.\
                GetConfigurationSpecificationFromType(
                        openravepy.IkParameterizationType.Transform6D,
                        'linear')
            traj.Init(spec)
            traj.Insert(traj.GetNumWaypoints(),
                        openravepy.poseFromMatrix(start_pose))
            traj.Insert(traj.GetNumWaypoints(),
                        openravepy.poseFromMatrix(goal_pose))
            # Not required
            #openravepy.planningutils.RetimeAffineTrajectory(
            #    traj,
            #    maxvelocities=0.1*numpy.ones(7),
            #    maxaccelerations=0.1*numpy.ones(7)
            #)

        return self.PlanWorkspacePath(robot, traj, timelimit)
    """


    @PlanningMethod
    def PlanToEndEffectorOffset(self, robot, direction,
                                distance,
                                threshold,
                                matching_fn_class,
                                distance_metric_fn,
                                ignore_segments=True,
                                max_recursion_depth=10,
                                terminate_on_max_recursion=True,
                                **kwargs):
        """
        """

        print "in RecursiveSnap: PlanToEndEffectorOffset() \n"

        if distance <= 0.0:
            raise ValueError("Distance must be non-negative.")
        elif numpy.linalg.norm(direction) == 0:
            raise ValueError("Direction must be non-zero")

        # Normalize the direction vector.
        direction = numpy.array(direction, dtype='float')
        direction /= numpy.linalg.norm(direction)

        with robot:
            manip = robot.GetActiveManipulator()
            start_pose = manip.GetEndEffectorTransform()
            traj = openravepy.RaveCreateTrajectory(self.env, '')
            spec = openravepy.IkParameterization.\
                            GetConfigurationSpecificationFromType(
                                 openravepy.IkParameterizationType.Transform6D,
                                 'linear')
            traj.Init(spec)
            traj.Insert(traj.GetNumWaypoints(),
                        openravepy.poseFromMatrix(start_pose))
            min_pose = numpy.copy(start_pose)
            min_pose[0:3, 3] += distance*direction
            traj.Insert(traj.GetNumWaypoints(),
                        openravepy.poseFromMatrix(min_pose))
            #if max_distance is not None:
            #    max_pose = numpy.copy(start_pose)
            #    max_pose[0:3, 3] += max_distance*direction
            #    traj.Insert(traj.GetNumWaypoints(),
            #                openravepy.poseFromMatrix(max_pose))

        return self.PlanWorkspacePath(robot, traj,
                                      threshold,
                                      matching_fn_class,
                                      distance_metric_fn,
                                      ignore_segments,
                                      max_recursion_depth,
                                      terminate_on_max_recursion,
                                      **kwargs
                                      )


    @PlanningMethod
    def PlanWorkspacePath(self, robot, workspace_path,
                          threshold,
                          matching_fn_class,
                          distance_metric_fn,
                          ignore_segments=True,
                          max_recursion_depth=10,
                          terminate_on_max_recursion=True,
                          **kwargs):
        """
        Plan a configuration space path given a workspace path.

        @param openravepy.robot robot: The robot object.
        @param openravepy.Trajectory workspace_path: An OpenRAVE workspace path,
                                                     being of some openravepy.IkParameterization type.
                                                     Timing is not required.
        @param float threshold: Maximum deviation from desired workspace path.
        @param class matching_fn_class: A class for matching points from two discrete sets, that
                                        exposes a method findMaxDeviation()
        @param function distance_metric_fn A function for computing the distance between two 4x4 transforms.
        @param bool ignore_segments: If True, the planner will do jointspace
                                     interpolation between the first and last
                                     waypoints.
                                     If False, the planner will process each
                                     segment individually, starting by doing
                                     jointspace interpolation between the
                                     first and second waypoints.
        @param float max_recursion_depth: The maximum recursion level to stop,
                                          in case a large joint configuration
                                          change means the planner is unable
                                          to satisfy the deviation bound.
        @param bool terminate_on_max_recursion: If True, the planner will raise
                                                an exception if the max
                                                recursion depth is reached.
                                                If False, the planner will
                                                ignore the segment that did
                                                not satisfy the deviation bound.
        @param dictionary kwargs: Optional arguments: rviz_tools_class: A class for publishing Rviz Markers.

        @returns openravepy.Trajectory jointspace_path: A OpenRAVE configuration space path.
        """

        def _checkCollisions(q_array):

            # Convert array of joint configurations into OpenRAVE trajectory
            cspec = self.robot.GetActiveConfigurationSpecification('linear')

            temp_traj = CreateJointSpaceTrajectory(robot, q_array)

            # Get joint configurations to check
            # Note: this returns a python generator, and if the
            # trajectory only has one waypoint then only the
            # start configuration will be collision checked.
            #
            # Sampling function:
            # 'Van der Corput'
            vdc = prpy.util.VanDerCorputSampleGenerator
            checks = prpy.util.GetLinearCollisionCheckPts(robot, \
                                                          temp_traj, \
                                                          norm_order=2, \
                                                          sampling_func=vdc)
            for t, q in checks:
                # Set the joint positions
                # Note: the planner is already using a cloned 'robot' object
                self.robot.SetActiveDOFValues(q)

                self.num_collision_checks += 1

                # Check for collisions
                report = openravepy.CollisionReport()
                if self.env.CheckCollision(robot, report=report):
                    raise CollisionPlanningError.FromReport(report)
                elif self.robot.CheckSelfCollision(report=report):
                    raise SelfCollisionPlanningError.FromReport(report)

        def _plan(q0, q1,
                  workspace_traj_t0=None,
                  workspace_traj_t1=None,
                  count=1):
            """
            Recursive function to do linear joint interpolation between two
            sample points of a workspace trajectory.
            The joint configurations at the beginning and end of each
            segment are also passed-in, to ensure continuity.

            @param array q0 Joint configuration at beginning of segment.
            @param array q1 Joint configuration at the end of segment.
            @param float workspace_traj_t0 Time along trajectory at
                                           beginning of segment.
            @param float workspace_traj_t1 Time along trajectory at
                                           the end of segment.
            @param int count: Counter used to count the max recursion
                              depth, starting at 1.

            @returns numpy.array q_array: An array where each row is a joint
                                          configuration.
            """

            if workspace_traj_t0 == None:
                workspace_traj_t0 = 0.0
            if workspace_traj_t1 == None:
                workspace_traj_t1 = self.workspace_traj.GetDuration()

            if workspace_traj_t1 <= workspace_traj_t0:
                raise ValueError("t1 must be greater than t0 in _plan()")

            if count == self.recursion_limit:
                # Reached maximum recursion depth,
                if self.terminate_on_max_recursion:
                    raise ValueError("Planner reached max_recursion_depth={:d}"
                                     " and failed to satisfy the deviation"
                                     " bound for workspace trajectory between"
                                     " t = {:.6f} to {:.6f} ".format(count, \
                                         workspace_traj_t0, workspace_traj_t1))
                else:
                    print "\nWARNING: "
                    print "Reached max_recursion_depth =", count, ", " \
                          "returning end-points for this segment."
                    q_array = numpy.vstack((q0, q1))
                    _checkCollisions(q_array)
                    return q_array
            count = count + 1

            # Count the deepest level of recursion reached
            if count > self.deepest_recursion:
                self.deepest_recursion = count


            # ToDo: can we chose the number of discretization samples automatically?
            num_samples = 10

            # Sample this segment of the workspace trajectory into a numpy
            # array of end effector transforms
            (P_times,workspace_traj_array) = \
                        SampleWorkspaceTrajectoryIntoArray(robot,
                                                           self.workspace_traj,
                                                           workspace_traj_t0,
                                                           workspace_traj_t1,
                                                           num_samples)

            # Linearly interpolate between two C-space configurations
            # (a SnapPlan without collision-checking)
            # and sample into a numpy array of transforms.
            snap_traj = \
                CreateWorkspaceTrajectoryByLinearInterpBetweenConfigs(self.env,
                                                                      robot,
                                                                      q0, q1)

            (Q_times,snap_traj_array) = \
                    SampleWorkspaceTrajectoryIntoArray(robot,
                                                       snap_traj,
                                                       0.0,
                                                       snap_traj.GetDuration(),
                                                       num_samples)

            if snap_traj.GetDuration() <= 0.0:
                # This should never happen
                raise ValueError("Fatal error: snap_traj has no duration")

            # Calculate the position of maximum deviation between the
            # desired workspace path and the path created by jointspace
            # interpolation
            # Result is: [path1_index, path2_index, distance]
            MFC = self.matching_fn_class(workspace_traj_array,
                                         snap_traj_array,
                                         self.distance_metric_fn)
            max_deviation_couple = MFC.findMaxDeviation()
            max_deviation = max_deviation_couple[2]

            # Debug:
            # Draw Rviz Markers
            if self.rviz_tools_class != None:
                # Draw the desired workspace path as
                # a series of pose axes
                PublishWorkspaceTrajectoryPoses(self.rviz_tools_class,
                                                robot,
                                                self.workspace_traj,
                                                n=30,
                                                axis_length=0.02,
                                                axis_radius=0.001)

                #P_list = Convert3DArrayToListOfArrays(workspace_traj_array)
                #self.rviz_tools_class.publishPath(P_list, 'purple', 0.002, None)

                # Draw the linearly interpolated trajectory
                Q_list = Convert3DArrayToListOfArrays(snap_traj_array)
                self.rviz_tools_class.publishPath(Q_list, 'purple', 0.002, None)

                # Draw the segments of the desired workspace path
                if self.ignore_traj_segments:
                    # Draw the entire workspace trajectory
                    PublishWorkspaceTrajectorySegments(self.rviz_tools_class,
                                                       robot,
                                                       self.workspace_traj,
                                                       'white', 0.005)
                else:
                    # Draw only the current workspace trajectory
                    # segment being processed
                    PublishWorkspaceTrajectoryClosestSegment(
                                                         self.rviz_tools_class,
                                                         robot,
                                                         self.workspace_traj,
                                                         'white', 0.005,
                                                         workspace_traj_t0,
                                                         workspace_traj_t1)

                # Draw all the matching lines between the workspace path
                # and the linearly interpolated trajectory
                for i in range(0, num_samples):
                    pt1 = workspace_traj_array[i]
                    pt2 = snap_traj_array[i]
                    duration = None
                    self.rviz_tools_class.publishLine(pt1, pt2,
                                                      'red', 0.0005, duration)
                # Draw the position of max deviation (in red)
                P_i = int(max_deviation_couple[0]) # - 1
                P_couple_point = workspace_traj_array[ P_i ]
                Q_i = int(max_deviation_couple[1]) # - 1
                Q_couple_point = snap_traj_array[ Q_i ]
                duration = None
                self.rviz_tools_class.publishLine(P_couple_point,
                                                  Q_couple_point,
                                                  'red', 0.003, duration)

                time.sleep(2.0)
                self.rviz_tools_class.deleteAllMarkers()
            
            if (max_deviation > self.max_deviation_threshold):
                # Max deviation exceeds threshold, split this segment of
                # the workspace trajectory into two segments:

                # Get the path indices at the position of max deviation
                P_i = int(max_deviation_couple[0])
                Q_i = int(max_deviation_couple[1])

                # Get the end-effector transforms at
                # the position of max deviation
                P_couple_point = workspace_traj_array[P_i]
                Q_couple_point = snap_traj_array[Q_i]

                # Get the times at the position of max deviation
                P_couple_time = P_times[P_i]
                Q_couple_time = Q_times[Q_i]

                # Get an IK solution for the end effector transform on
                # the workspace path at the position of max deviation
                filter_options = openravepy.IkFilterOptions.CheckEnvCollisions
                manipulator = robot.GetActiveManipulator()
                T_ee_new = P_couple_point
                q_new = None
                with robot: # save robot state
                    # Bias the IK search
                    self.robot.SetActiveDOFValues((q0+q1)/2.0)
                    # Bias the IK search to the joint configuration at the
                    # start or end of this segment:
                    #robot.SetActiveDOFValues( q0 )
                    #robot.SetActiveDOFValues( q1 )

                    # releasegil=True ==> don't block on the C++ call
                    q_new = manipulator.FindIKSolution(T_ee_new,
                                                       filter_options,
                                                       ikreturn=False,
                                                       releasegil=True
                                                       )
                    if q_new == None: 
                        raise ValueError("There is no IK solution " \
                                         "for T_ee_new")

                # First segment, before couple point:

                # By default, return the joint configurations at the
                # start and end of this segment
                q_segment1 = numpy.vstack((q0, q_new))

                # If this segment is long enough, recursively call the
                # planner again
                if any(abs(q0 - q_new) > self.joint_resolutions):
                    q_segment1 = _plan(q0, q_new, workspace_traj_t0, P_couple_time, count)
                else:
                    print "WARNING: Joint steps are below DOF resolution"
                    q_array = numpy.vstack((q0, q_new))
                    _checkCollisions(q_array)

                # Second segment, after the couple point:

                # By default, return the joint configurations at the
                # start and end of this segment
                q_segment2 = numpy.vstack((q_new, q1))

                # If this segment is long enough, recursively call the
                # planner again
                if any( abs(q_new - q1) > self.joint_resolutions ):
                    q_segment2 = _plan(q_new, q1, P_couple_time, workspace_traj_t1, count)
                else:
                    print "WARNING: Joint steps are below DOF resolution"
                    q_array = numpy.vstack((q_new, q1))
                    _checkCollisions(q_array)

                # Stack the joint configurations from both segments into
                # one matrix, without duplication:
                try:
                    q_seg1_num_rows = numpy.shape(q_segment1)[0]
                except:
                    raise ValueError("Fatal error: failed to get shape "
                                     " of array 'q_segment1' ")
                try:
                    q_seg2_num_rows = numpy.shape(q_segment2)[0]
                except:
                    raise ValueError("Fatal error: failed to get shape "
                                     " of array 'q_segment2' ")
                total_len_q_segs = q_seg1_num_rows + q_seg2_num_rows
                q_array = numpy.zeros((total_len_q_segs,7))
                # use all rows of q_segment1
                q_segment2 = q_segment2[1:,:]
                q_array = numpy.vstack((q_segment1, q_segment2))

            else:
                # This segment satisfies the desired deviation
                q_array = numpy.vstack((q0, q1))
                _checkCollisions(q_array)

            return q_array


        print "in RecursiveSnap: PlanWorkspacePath()"
        print "num waypoints = ", workspace_path.GetNumWaypoints()


        # ToDo: fix AngleBetweenQuaternions() in here

        # Compute path-length timing of workspace trajectory
        self.workspace_traj = \
                            prpy.util.ComputeGeodesicUnitTiming(workspace_path)

        self.robot = robot

        # Get the resolution (in radians) for each joint
        DOF_resolution_scale = 0.25
        self.joint_resolutions = self.robot.GetActiveDOFResolutions() * \
                                                           DOF_resolution_scale

        # Max allowable 'distance' between desired workspace path
        # and the Snap path
        self.max_deviation_threshold = threshold

        self.matching_fn_class = matching_fn_class
        self.distance_metric_fn = distance_metric_fn

        self.ignore_traj_segments = ignore_segments
        self.recursion_limit = max_recursion_depth
        self.terminate_on_max_recursion = terminate_on_max_recursion

        # Get any optional keyword arguments
        self.rviz_tools_class = kwargs.get('rviz_tools_class', None)

        # For recording statistics
        self.deepest_recursion = 0
        self.num_collision_checks = 0

        # Get current joint configuration
        q_start = robot.GetActiveDOFValues()

        # Get a joint configuration for the end effector transform at
        # the end of the workspace trajectory
        P = self.workspace_traj.GetWaypoint(-1)[0:7] # first 7 values are pose
        T_goal = openravepy.matrixFromPose(P)
        filter_options = openravepy.IkFilterOptions.CheckEnvCollisions
        manipulator = robot.GetActiveManipulator()
        q_goal = manipulator.FindIKSolution(T_goal, filter_options)
        if q_goal == None:
            raise ValueError("There is no IK solution to goal pose")

        # Draw the segments of the desired workspace path
        if self.rviz_tools_class != None:
            PublishWorkspaceTrajectorySegments(self.rviz_tools_class,
                                               robot, self.workspace_traj,
                                               color='white', line_width=0.005)

        if self.ignore_traj_segments:
            # Treat the entire workspace trajectory as a single segment,
            # the planner will initially do jointspace interpolation
            # from q_start to q_goal
            q_array = _plan(q_start, q_goal)
        else:
            # Process each segment of the workspace
            # trajectory separately
            q_array = None
            q_prev = q_start
            waypoint_times = \
                       GetWorkspaceTrajectoryWaypointTimes(self.workspace_traj)
            filter_options = openravepy.IkFilterOptions.CheckEnvCollisions
            manipulator = robot.GetActiveManipulator()

            num_waypoints = self.workspace_traj.GetNumWaypoints()
            t_prev = 0.0
            for i in xrange(1, num_waypoints):
                P = self.workspace_traj.GetWaypoint(i)[0:7] # 7 values are pose
                T_ee = openravepy.matrixFromPose(P)
                t = waypoint_times[i]

                q = manipulator.FindIKSolution(T_ee, filter_options)
                if q == None:
                    raise ValueError("There is no IK solution for waypoint "
                                     "{:d} ".format(i) )

                q_segment = _plan(q_prev, q, t_prev, t)

                if i == 1:
                    q_array = q_segment
                else:
                    # Append the new joint configurations, except the first
                    # row which would be a duplication
                    q_array = numpy.vstack((q_array, q_segment[1:,:]))

                q_prev = q
                t_prev = t

        # Convert array of joint configurations into OpenRAVE trajectory
        cspec = robot.GetActiveConfigurationSpecification('linear')
        jointspace_path = CreateJointSpaceTrajectory(robot, q_array)

        """
        # Debug:
        # Collision-check the entire path again
        vdc = prpy.util.VanDerCorputSampleGenerator
        checks = prpy.util.GetLinearCollisionCheckPts(robot, \
                                                      jointspace_path, \
                                                      norm_order=2, \
                                                      sampling_func=vdc)
        count = 0
        for t, q in checks:
            # Set the joint positions
            # Note: the planner is already using a cloned 'robot' object
            robot.SetActiveDOFValues(q)
            count += 1
            # Check for collisions
            report = openravepy.CollisionReport()
            if self.env.CheckCollision(robot, report=report):
                raise CollisionPlanningError.FromReport(report)
            elif robot.CheckSelfCollision(report=report):
                raise SelfCollisionPlanningError.FromReport(report)
        print "did", count, "collision checks."
        """

        print "\nDone! \n"
        #print "Joint-space path: "
        #print q_array
        print("No. of waypoints = %i " % jointspace_path.GetNumWaypoints())
        print("Deepest recursion level = %i " % self.deepest_recursion)
        print("No. of collision checks = %i " % self.num_collision_checks)
        print("")

        # Tag the trajectory, this is required by the Base Class
        SetTrajectoryTags(jointspace_path, {Tags.CONSTRAINED: True}, append=True)

        return jointspace_path
