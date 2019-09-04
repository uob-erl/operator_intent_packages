#!/usr/bin/env python
#from __future__ import division, print_function



# simple code -- bayesian estimation with 2 obs (eucl & angle)
# assusmption : All goals are located in front of us. No hidden goals behind walls.

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
# from filterpy.discrete_bayes import normalize
# from filterpy.discrete_bayes import update




x_robot = 0.0
y_robot = 0.0
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
Gprime = Point()
G1 = Point()
G2 = Point()
G3 = Point()
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
    global x_robot, y_robot, qx, qy, qz, qw
    x_robot = mes.pose.pose.position.x
    y_robot = mes.pose.pose.position.y
    qx = mes.pose.pose.orientation.x
    qy = mes.pose.pose.orientation.y
    qz = mes.pose.pose.orientation.z
    qw = mes.pose.pose.orientation.w

# -------------------------------------------------- C A L L B A C K S --------------------------------------------------------------- #




# -------------------------------------------------- F U N C T I O N S --------------------------------------------------------------- #

# compute likelihood : P(obs|goal) = normalized [e^(-k*obs)] , for all goals
def compute_like(dist, Angle, mu, beta):
    out0 = np.exp(-beta * dist) * np.exp(-mu * Angle)
    like = out0 / np.sum(out0)
    return like


# compute conditional : normalized [Sum P(goal.t|goal.t-1) * b(goal.t-1)]
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


    # create tf.TransformListener objects
    listener = tf.TransformListener()
    listenerNAV = tf.TransformListener()
    listener1 = tf.TransformListener()
    listener2 = tf.TransformListener()
    listener3 = tf.TransformListener()


    # subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)


    # publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size=1)


    # declare some variables
    mu = 0.1
    beta = 0.3
    n = 4   # number of goals
    l = 0.75
    Delta = 0.2

    # Initialize Prior-beliefs according to goals' number
    # nav goal should have higher prior belief at t=0
    data_prior = np.ones(n-1) * (1-l)/(n-1)   # P(gnav)=0.75 , P(g0)= ... , P(g1)= ...
    prior = data_prior
    prior = np.insert(prior, 0, l, axis=0)


    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data_cpt = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data_cpt, 1-Delta)
    cond = data_cpt


    rate=rospy.Rate(10)



    while not rospy.is_shutdown():

        # robot coordinates (MAP FRAME)
        robot_coord = [x_robot, y_robot]

        g_prime = [x_nav, y_nav]  # CLICKED point - g'
        g1 = [-2.83128278901, -3.62014215797]
        g2 = [-0.215494300143, -5.70071558441]
        g3 = [3.08670737031, -5.93227324436]

        # goals coordinates (MAP FRAME)
        targets = [g_prime, g1, g2, g3]


# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #

        # prepare transformation from g_prime(MAP FRAME) to gprime -> g_prime_new(ROBOT FRAME)
        Gprime_msg = PointStamped()
        Gprime_msg.header.frame_id = "map"
        Gprime_msg.header.stamp = rospy.Time(0)
        Gprime_msg.point.x = g_prime[0]
        Gprime_msg.point.y = g_prime[1]

        # prepare transformation from g0(MAP FRAME) to g1 -> g1_new(ROBOT FRAME)
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

        # prepare transformation from g1(MAP FRAME) to g3 -> g3_new(ROBOT FRAME)
        G3_msg = PointStamped()
        G3_msg.header.frame_id = "map"
        G3_msg.header.stamp = rospy.Time(0)
        G3_msg.point.x = g3[0]
        G3_msg.point.y = g3[1]




        try:

            (trans, rot) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", Gprime_msg)  # transform g_prime to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y


        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):

            continue


        # convert from quaternion to RPY (the yaw of the robot in /map frame)
        orientation_list = rot

        # rotation angle of the robot = yaw in ROBOT FRAME
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        yaw_degrees = yaw * 180 / np.pi
        rospy.loginfo("yaw: %s", yaw_degrees)


        # declare NEW coordinates' goals after the transformations (ROBOT FRAME)
        g_prime_new = [list_nav.point.x, list_nav.point.y]
        g1_new = [list1.point.x, list1.point.y]
        g2_new = [list2.point.x, list2.point.y]
        g3_new = [list3.point.x, list3.point.y]


        # NEW robot's coordinates after transformation (we don't care !) --- robot's x,y always 0 in ROBOT FRAME
        # robot = [trans[0], trans[1]]
        # rospy.loginfo("Robot_FRAME: %s", robot)




        # goals coordinates (ROBOT FRAME)
        new_goals = [g_prime_new[0], g_prime_new[1], g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]]
        new = np.array(new_goals)

# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #




# -------------------------------------------------- O B S E R V A T I O N S ------------------------------------------------------------- #

        # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
        measure = np.array([])
        for p in targets:
            dist = distance.euclidean(robot_coord, p)
            measure = np.append(measure, dist)
        dist = measure

        # angles' computation between robot (x=0, y=0) & each goal (2nd Observation)
        robot_base = [0, 0]

        # # if n=3 ..
        # ind_pos_x = [0, 2, 4]
        # ind_pos_y = [1, 3, 5]

        # if n=4 ..
        ind_pos_x = [0, 2, 4, 6]
        ind_pos_y = [1, 3, 5, 7]

        # # if n=5 ..
        # ind_pos_x = [0, 2, 4, 6, 8]
        # ind_pos_y = [1, 3, 5, 7, 9]

        # and so on ...

        dx = new - robot_base[0]
        dy = new - robot_base[1]
        Dx = dx[ind_pos_x]
        Dy = dx[ind_pos_y]
        angle = np.arctan2(Dy, Dx) * 180 / np.pi
        Angle = abs(angle)

# -------------------------------------------------- O B S E R V A T I O N S ------------------------------------------------------------- #




# -------------------------------------------------- B A Y E S   R U L E ------------------------------------------------------------- #

        # BAYES' FILTER
        likelihood = compute_like(dist, Angle, mu, beta)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        index = np.argmax(posterior)
        prior = posterior

# -------------------------------------------------- B A Y E S   R U L E ------------------------------------------------------------- #




        # print ...
        rospy.loginfo("NAV_goal: %s", g_prime)
        rospy.loginfo("Distance: %s", dist)
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
