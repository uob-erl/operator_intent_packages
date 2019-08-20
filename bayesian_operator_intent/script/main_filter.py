#!/usr/bin/env python
#from __future__ import division, print_function


# main filter -- bayesian estimation with 2 observations ( 3 in total --> 2 after combining)


import rospy
import numpy as np
import random
import math
import tf
import actionlib
from actionlib import ActionServer
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial import distance
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from random import randint




x0 = 0.0
y0 = 0.0
rot_angle = 0
qx = 0
qy = 0
qz = 0
qw = 0
x_nav = 0
y_nav = 0
qx_nav = 0
qy_nav = 0
qz_nav = 0
qw_nav = 0
roll = 0
pitch = 0
yaw = 0
delta_yaw = 0
orientation_list = 0
path_length = 0
gprime = Point()
G0 = Point()
G1 = Point()
G2 = Point()
G4 = Point()
G9 = Point()
G10 = Point()
G11 = Point()
Goal = PoseStamped()
Start = PoseStamped()




# -------------------------------------------------- C A L L B A C K S --------------------------------------------------------------- #

# callback function for navigational goal (g')
def call_nav(sms):
    global x_nav, y_nav, qx_nav, qy_nav, qz_nav, qw_nav
    x_nav = sms.pose.position.x
    y_nav = sms.pose.position.y
    qx_nav = sms.pose.orientation.x
    qy_nav = sms.pose.orientation.y
    qz_nav = sms.pose.orientation.z
    qw_nav = sms.pose.orientation.w


# callback function for robot's coordinates (MAP FRAME)
def call_rot(mes):
    global x0, y0, qx, qy, qz, qw
    x0 = mes.pose.pose.position.x
    y0 = mes.pose.pose.position.y
    qx = mes.pose.pose.orientation.x
    qy = mes.pose.pose.orientation.y
    qz = mes.pose.pose.orientation.z
    qw = mes.pose.pose.orientation.w


# callback function for evaluating path length of each goal
def call(path_msg):
    global path_length
    path_length = 0
    for i in range(len(path_msg.poses) - 1):
        position_a_x = path_msg.poses[i].pose.position.x
        position_b_x = path_msg.poses[i+1].pose.position.x
        position_a_y = path_msg.poses[i].pose.position.y
        position_b_y = path_msg.poses[i+1].pose.position.y

        path_length += np.sqrt(np.power((position_b_x - position_a_x), 2) + np.power((position_b_y - position_a_y), 2))

# -------------------------------------------------- C A L L B A C K S --------------------------------------------------------------- #




# -------------------------------------------------- F U N C T I O N S --------------------------------------------------------------- #

# compute likelihood : P(obs|goal) = normalized [e^(-k*obs)] , for all goals .. "SENSOR model"
def compute_like(dist, Angle, k, m):
    out0 = np.exp(-k * dist) * np.exp(-m * Angle)
    like = out0 / np.sum(out0)
    return like


# compute conditional : normalized [Sum P(goal.t|goal.t-1) * b(goal.t-1)] .. "ACTION model"
def compute_conditional(cond, prior):
    out1 =  np.matmul(cond, prior.T)
    sum = out1 / np.sum(out1)
    return sum


# compute posterior P(goal|obs) = normalized(likelihood * conditional)
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)
    return post

# -------------------------------------------------- F U N C T I O N S --------------------------------------------------------------- #




def run():

    rospy.init_node('bayesian_filter')

    # Create a callable proxy to GetPlan service
    get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)


    # create tf.TransformListener objects
    listener = tf.TransformListener()
    listenerNAV = tf.TransformListener()
    listener0 = tf.TransformListener()
    listener1 = tf.TransformListener()
    listener2 = tf.TransformListener()
    listener4 = tf.TransformListener()
    listener9 = tf.TransformListener()
    #listener10 = tf.TransformListener()
    listener11 = tf.TransformListener()


    # Subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)


    # Publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size=1)


    # declare variables
    k = 0.3
    m = 0.1
    n = 4   # number of total goals (prime+subgoals)
    l = 0.75
    Delta = 0.2


    # Initialize Prior-beliefs according to goals' number .. nav goal should have higher prior belief at t=0
    data0 = np.ones(n-1) * (1-l)/(n-1)   # P(gnav)=0.75 , P(g0)= ... , P(g1)= ...
    prior = data0
    prior = np.insert(prior, 0, l, axis=0)


    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data1 = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data1, 1-Delta)
    cond = data1


    rate = rospy.Rate(8) # 5 Hz (5 loops/sec) .. (0.2 sec)


    while not rospy.is_shutdown():


        #offset = 0.25
        # robot coordinates (MAP FRAME)
        robot_coord = [x0, y0]

        # goals coordinates (MAP FRAME)
        g_prime = [x_nav, y_nav]  # CLICKED point - g'
        g0 = [-2.83128278901, -3.62014215797]
        g1 = [-0.215494300143, -5.70071558441]
        g2 = [3.08670737031, -5.93227324436]
        # g3 = [6.94527384787, -11.717640655]
        g4 = [8.51079199583, 4.698199058]
        #g5 = [9.00735322774, -0.677194482712]
        #g6 = [11.8716085357, -0.506166845716]
        #g7 = [14.9899956087, 5.20262059311]
        #g8 = [20.101826819, -8.62395824827]
        g9 = [22.9649931871, 10.4371948525]
        #g10 = [25.5818735158, 15.74648043]
        g11 = [27.8554619121, 6.78231736479]


        targets_FIRST = [g_prime, g0, g1, g2] # list of FIRST set of goals (MAP FRAME) --> useful for euclidean distance
        targets_SECOND = [g_prime, g4, g9, g11] # list of SECOND set of goals (MAP FRAME) --> useful for euclidean distance


        # FIRST set with offset --> useful for GetPlan
        g0_off = [-2.66417694092, -3.15912365913]
        g1_off = [-0.00289916992188, -5.19672012329]
        g2_off = [3.34997367859, -5.3837480545]
        targets_FIRST_off = [g_prime, g0_off, g1_off, g2_off]

        # SECOND set with offset --> useful for GetPlan
        g4_off = [8.9230594635, 4.62477874756]
        g9_off = [22.652551651, 9.93627738953]
        g11_off = [27.10601425171, 6.8376455307]
        targets_SECOND_off = [g_prime, g4_off, g9_off, g11_off]




# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #

        # prepare transformation from g_prime(MAP FRAME) to gprime -> g_prime_new(ROBOT FRAME)
        gprime_msg = PointStamped()
        gprime_msg.header.frame_id = "map"
        gprime_msg.header.stamp = rospy.Time(0)
        gprime_msg.point.x = g_prime[0]
        gprime_msg.point.y = g_prime[1]

        # prepare transformation from g0(MAP FRAME) to g0 -> g0_new(ROBOT FRAME)
        G0_msg = PointStamped()
        G0_msg.header.frame_id = "map"
        G0_msg.header.stamp = rospy.Time(0)
        G0_msg.point.x = g0[0]
        G0_msg.point.y = g0[1]

        # prepare transformation from g1(MAP FRAME) to g1 -> g1_new(ROBOT FRAME)
        G1_msg = PointStamped()
        G1_msg.header.frame_id = "map"
        G1_msg.header.stamp = rospy.Time(0)
        G1_msg.point.x = g1[0]
        G1_msg.point.y = g1[1]

        # prepare transformation from g2(MAP FRAME) to g2 -> g2_new(ROBOT FRAME)
        G2_msg = PointStamped()
        G2_msg.header.frame_id = "map"
        G2_msg.header.stamp = rospy.Time(0)
        G2_msg.point.x = g2[0]
        G2_msg.point.y = g2[1]

        # prepare transformation from g2(MAP FRAME) to g2 -> g2_new(ROBOT FRAME)
        G4_msg = PointStamped()
        G4_msg.header.frame_id = "map"
        G4_msg.header.stamp = rospy.Time(0)
        G4_msg.point.x = g4[0]
        G4_msg.point.y = g4[1]

        # prepare transformation from g2(MAP FRAME) to g9 -> g9_new(ROBOT FRAME)
        G9_msg = PointStamped()
        G9_msg.header.frame_id = "map"
        G9_msg.header.stamp = rospy.Time(0)
        G9_msg.point.x = g9[0]
        G9_msg.point.y = g9[1]
        #
        # # prepare transformation from g2(MAP FRAME) to g10 -> g10_new(ROBOT FRAME)
        # G10_msg = PointStamped()
        # G10_msg.header.frame_id = "map"
        # G10_msg.header.stamp = rospy.Time(0)
        # G10_msg.point.x = g10[0]
        # G10_msg.point.y = g10[1]
        #
        # prepare transformation from g2(MAP FRAME) to g11 -> g11_new(ROBOT FRAME)
        G11_msg = PointStamped()
        G11_msg.header.frame_id = "map"
        G11_msg.header.stamp = rospy.Time(0)
        G11_msg.point.x = g11[0]
        G11_msg.point.y = g11[1]


        try:

            (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", gprime_msg)
            list = listener0.transformPoint("/base_link", G0_msg)  # transform g0 to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg) # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg) # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list4 = listener4.transformPoint("/base_link", G4_msg)
            list9 = listener9.transformPoint("/base_link", G9_msg) # transform g9 to base_link (ROBOT FRAME) , returns x,y
            #list10 = listener10.transformPoint("/base_link", G10_msg) # transform g10 to base_link (ROBOT FRAME) , returns x,y
            list11 = listener11.transformPoint("/base_link", G11_msg) # transform g11 to base_link (ROBOT FRAME) , returns x,y

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):

            continue


        # convert from quaternion to RPY (the yaw of the robot in /map frame)
        orientation_list = rotation

        # rotation angle of the robot = yaw in ROBOT FRAME
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        yaw_degrees = yaw * 180 / np.pi



        # NEW coordinates' goals after transformations (ROBOT FRAME)
        g_prime_new = [list_nav.point.x, list_nav.point.y]
        g0_new = [list.point.x, list.point.y]
        g1_new = [list1.point.x, list1.point.y]
        g2_new = [list2.point.x, list2.point.y]
        g4_new = [list4.point.x, list4.point.y]
        g9_new = [list9.point.x, list9.point.y]
        #g10_new = [list10.point.x, list10.point.y]
        g11_new = [list11.point.x, list11.point.y]


        # NEW robot's coordinates after transformation (we don't care !) --- robot's x,y always 0 in ROBOT FRAME
        # robot = [translation[0], translation[1]]
        # rospy.loginfo("Robot_FRAME: %s", robot)




        # list of FIRST set of goals (ROBOT FRAME)
        new_goals_FIRST = [g_prime_new[0], g_prime_new[1], g0_new[0], g0_new[1], g1_new[0], g1_new[1], g2_new[0], g2_new[1]] # list
        new_FIRST = np.array(new_goals_FIRST) # array --> useful for angle computation


        # # list of SECOND set of goals (ROBOT FRAME)
        new_goals_SECOND = [g_prime_new[0], g_prime_new[1], g4_new[0], g4_new[1], g9_new[0], g9_new[1], g11_new[0], g11_new[1]] # list
        new_SECOND = np.array(new_goals_SECOND) # array --> useful for angle computation

# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #




# -------------------------------------------------- O B S E R V A T I O N S ------------------------------------------------------------- #

        # check euclidean distance(nav-g1) to define the proper set of goals (either 0,1,2 OR 4,9,11)
        check = distance.euclidean(g_prime, g0)
        rospy.loginfo("CHECK: %s", check)



        if check < 18:

            rospy.loginfo("status=0")

            # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
            measure = np.array([])
            for x in targets_FIRST:
                dist = distance.euclidean(robot_coord, x)
                measure = np.append(measure, dist)
            dist = measure
            rospy.loginfo("Distance: %s", dist)


            # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
            robot_base = [0, 0]
            ind_pos_x = [0, 2, 4, 6]
            ind_pos_y = [1, 3, 5, 7]
            dx = new_FIRST - robot_base[0]
            dy = new_FIRST - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)


            # generate plan towards goals --> n-path lengths .. (3rd Observation)
            PATH = np.array([])
            for j in targets_FIRST_off:

                Start = PoseStamped()
                Start.header.seq = 0
                Start.header.frame_id = "map"
                Start.header.stamp = rospy.Time(0)
                Start.pose.position.x = robot_coord[0]
                Start.pose.position.y = robot_coord[1]

                Goal = PoseStamped()
                Goal.header.seq = 0
                Goal.header.frame_id = "map"
                Goal.header.stamp = rospy.Time(0)
                Goal.pose.position.x = j[0]
                Goal.pose.position.y = j[1]

                srv = GetPlan()
                srv.start = Start
                srv.goal = Goal
                srv.tolerance = 0.5
                resp = get_plan(srv.start, srv.goal, srv.tolerance)
                rospy.sleep(0.02) # 0.02 x 4 = 0.08 sec

                PATH = np.append(PATH, path_length)
            path = PATH


        else:


            rospy.loginfo("status=1")

            # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
            measure = np.array([])
            for x in targets_SECOND:
                dist = distance.euclidean(robot_coord, x)
                measure = np.append(measure, dist)
            dist = measure
            rospy.loginfo("Distance: %s", dist)


            # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
            robot_base = [0, 0]
            ind_pos_x = [0, 2, 4, 6]
            ind_pos_y = [1, 3, 5, 7]
            dx = new_SECOND - robot_base[0]
            dy = new_SECOND - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)


            # generate plan towards goals --> n-path lengths .. (3rd Observation)
            PATH = np.array([])
            for j in targets_SECOND_off:

                Start = PoseStamped()
                Start.header.seq = 0
                Start.header.frame_id = "map"
                Start.header.stamp = rospy.Time(0)
                Start.pose.position.x = robot_coord[0]
                Start.pose.position.y = robot_coord[1]

                Goal = PoseStamped()
                Goal.header.seq = 0
                Goal.header.frame_id = "map"
                Goal.header.stamp = rospy.Time(0)
                Goal.pose.position.x = j[0]
                Goal.pose.position.y = j[1]

                srv = GetPlan()
                srv.start = Start
                srv.goal = Goal
                srv.tolerance = 0.5
                resp = get_plan(srv.start, srv.goal, srv.tolerance)
                rospy.sleep(0.02) # 0.02 x 4 = 0.08 sec

                PATH = np.append(PATH, path_length)
            path = PATH

# -------------------------------------------------- O B S E R V A T I O N S ---------------------------------------------------------- #




# -------------------------------------------------- C H E C K ---------------------------------------------------------- #

        # check the difference between euclidean distance & path length for each goal
        diff = abs(path - dist)

        value = (0.3 * sum(path)) / n

        result = any(q > value for q in diff)
        if result:
            rospy.loginfo("yes - BAYES with diff & angle") # bayes me gwnia k path
            dist = diff
            #k = 0.3
            #m = 0.1
        else:
            rospy.loginfo("no - BAYES with eucl & angle") # bayes me gwnia k dist
            dist = dist
            #k = 0.3
            #m = 0.1

# -------------------------------------------------- C H E C K  ---------------------------------------------------------- #




# -------------------------------------------------- B A Y E S'  R U L E --------------------------------------------------------------- #

        # BAYES' FILTER
        likelihood = compute_like(dist, Angle, k, m)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        index = np.argmax(posterior)
        prior = posterior

# -------------------------------------------------- B A Y E S'  R U L E --------------------------------------------------------------- #




        # print ...
        rospy.loginfo("rotate: %s", yaw_degrees)
        rospy.loginfo("len: %s", path)
        rospy.loginfo("OPERATOR_goal: %s", g_prime)
        rospy.loginfo("Difference: %s", diff)
        rospy.loginfo("Angles: %s", Angle)
        rospy.loginfo("Posterior: %s", posterior)
        rospy.loginfo("Potential Goal is %s", index)




        pub.publish(index)


        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
