#!/usr/bin/env python
#from __future__ import division, print_function

'''
This script estimates the operator's intent (i.e. most probable goal) using Recursive Bayesian Estimation.
Here is a better representation/version of the implementation. Each step of the process can be found in the corresponding function. The process is described as follows:
PROCESS: [ initializations -> prepare goals for transformation(function) -> get the new goals coordinates(function) -> get distance,path,angle(3 functions) ->
-> get BAYES estimation(3 functions) -> publish the most_probable_goal = INTENT ]

TODO(done): -- a class for BAYES similar to scripts I've created for offline computation&evaluation based on the extracted data texts of rosbags
      -- remove unnecessary imports
      -- minor changes to the code for being more flexible and attractive

alternative (back up) script for BOIR if anything goes wrong in 'main.py'
and 'class_bayes.py'
'''

import rospy
import numpy as np
import random
import math
import tf
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial import distance
from std_msgs.msg import Float32, Int8
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
import time


# -------------------------- ROS-based FUNCTIONS -------------------------- #

# callback function for robot's coordinates (MAP FRAME)
def call_robot(message):
    global x_robot, y_robot, qx, qy, qz, qw
    x_robot = message.pose.pose.position.x
    y_robot = message.pose.pose.position.y
    qx = message.pose.pose.orientation.x
    qy = message.pose.pose.orientation.y
    qz = message.pose.pose.orientation.z
    qw = message.pose.pose.orientation.w


# callback function for evaluating path length of each goal
# tip : calculate the distance of every two points and add them up
def call_len(path_msg):
    global path_length
    path_length = 0
    # loop through each adjacent pair of Poses in the path message
    for i in range(len(path_msg.poses) - 1):
        position_a_x = path_msg.poses[i].pose.position.x
        position_b_x = path_msg.poses[i+1].pose.position.x
        position_a_y = path_msg.poses[i].pose.position.y
        position_b_y = path_msg.poses[i+1].pose.position.y
        path_length += np.sqrt(np.power((position_b_x - position_a_x), 2) + np.power((position_b_y - position_a_y), 2))


def getPrepared():
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
    # prepare transformation from g3(MAP FRAME) to g3 -> g3_new(ROBOT FRAME)
    G3_msg = PointStamped()
    G3_msg.header.frame_id = "map"
    G3_msg.header.stamp = rospy.Time(0)
    G3_msg.point.x = g3[0]
    G3_msg.point.y = g3[1]
    return G1_msg, G2_msg, G3_msg


def get_processed_goals():
    orientation_list = rotation  # convert from quaternion to RPY (the yaw of the robot in /map frame)
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    yaw_degrees = yaw * 180 / np.pi
    rot = yaw_degrees  # rotation angle of the robot = yaw in ROBOT FRAME
    g1_new = [list1.point.x, list1.point.y]  # NEW coordinates' goals after transformations (ROBOT FRAME)
    g2_new = [list2.point.x, list2.point.y]  #  -//-
    g3_new = [list3.point.x, list3.point.y]  #  -//-
    new_goals = [g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]] # list of set of goals (ROBOT FRAME)
    new = np.array(new_goals) # array --> useful for angle computation
    return new


# --------------------------  FUNCTIONS -------------------------- #

def calculate_distance():
    robot_coord = [x_robot, y_robot]  # robot coordinates (MAP FRAME) they are given through call_robot callback
    measure = np.array([])
    for x in targets:
        dis = distance.euclidean(robot_coord, x)
        measure = np.append(measure, dis)
    dis = measure
    return dis


def calculate_path():
    robot_coord = [x_robot, y_robot]  # robot coordinates (MAP FRAME) they are given through call_robot callback
    length = np.array([])
    for j in targets:
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
        rospy.sleep(0.05) # 0.05 x 3 = 0.15 sec

        length = np.append(length, path_length)
    return length


def calculate_angle(): # assume that robot's base is always (0,0)
    xs = []
    ys = []
    it = iter(new)
    [(xs.append(i), ys.append(next(it))) for i in it]
    angles = np.arctan2(ys, xs) * 180 / np.pi
    Angle = abs(angles)
    print("Angles: ", Angle)
    return Angle


# BAYES: step 1   ~ take two observations into account (Angle & Path)
def compute_likelihood():
    a = Angle / maxA
    p = path / maxP
    c_l = np.exp(-a / wA) * np.exp(-p / wP)
    return c_l

# BAYES: step 2
def compute_conditional():
    c_c = np.matmul(transition, prior.T)
    return c_c

# BAYES: step 3
def compute_posterior():
    out2 = likelihood * summary
    c_p = out2 / np.sum(out2)
    return c_p


# MAIN
if __name__=='__main__':
    try:
        # --------------------------- ROS necessary stuff --------------------------- #
        rospy.init_node('test_fcn')
        get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)
        sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call_len)
        robot_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_robot)
        intent_pub = rospy.Publisher('most_probable_goal', Int8, queue_size = 1)
        listener = tf.TransformListener()  # create tf.TransformListener objects
        listener1 = tf.TransformListener()
        listener2 = tf.TransformListener()
        listener3 = tf.TransformListener()

        # --------------------------- INITIALIZE PARAMETERS --------------------------- #
        g1 = [24.4425258636, 12.7283153534] # g left  (known a-priori)
        g2 = [29.08319664, 12.852309227]    # g center = HUMAN  (known a-priori)
        g3 = [33.7569503784, 12.5955343246] # g right  (known a-priori)
        targets = [g1, g2, g3]
        x_robot, y_robot = 0, 0
        qx, qy, qz, qw = 0, 0, 0, 0
        path_length, index = 0, 0
        wA = 0.6             # weighting factor for Angle
        wP = 0.4             # weighting factor for path
        maxA = 180           # max value the angle can take
        maxP = 25            # max value the path can take (depends on the arena)
        n_goals = 3          # number of total goals
        Delta = 0.2          # factor that determines CPT below

        # --------------------------- INITIALIZE 'PRIOR BELIEF' & 'CPT' --------------------------- #
        prior = np.ones(n_goals) * 1/n_goals   # Initialize Prior-beliefs according to goals' number (e.g. P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33  )
        data_cpt = np.ones((n_goals, n_goals)) * (Delta / (n_goals-1))
        np.fill_diagonal(data_cpt, 1-Delta)
        transition = data_cpt                  # creation of Conditinal Probability Table 'nxn' according to goals & Delta

        done = rospy.is_shutdown()
        rate = rospy.Rate(4)

# -------------------------------  MAIN LOOP ------------------------------- #
        while not done:
            start_time = time.time()  # is used to give us loop's execution time
            G1_msg, G2_msg, G3_msg = getPrepared()  # prepare goals for transformation
            try:
                (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
                list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
                list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
                list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            # --------------------------  GET USEFUL OBSERVATIONS  -------------------------- #
            new = get_processed_goals()      # get the new goals (i.e. new coordinates) in ROBOT FRAME
            Distance = calculate_distance()  # get Euclidean distance between robot and each goal (~optional)
            path = calculate_path()          # get path distance between robot and each goal
            Angle = calculate_angle()        # get angle between robot and each goal

            # --------------------------  PERFORM BAYESIAN ESTIMATION STEPS -------------------------- #
            #rospy.loginfo("prior: %s", prior)  # in the first iteration should be 1/n_goals (e.g. if n_goals=3 then prior = [0.33, 0.33, 0.33])
            likelihood = compute_likelihood()  # get likelihood P(Z|g) where Z[angle, path] for each goal g
            summary = compute_conditional()    # get result from CPT and prior belief
            posterior = compute_posterior()    # get posterior belief P(g|Z)
            index = np.argmax(posterior)       # get the most probable goal
            prior = posterior                  # set posterior as prior for the next iteration

            # print ..
            #rospy.loginfo("Distance: %s", Distance)
            #rospy.loginfo("Path: %s", path)
            #rospy.loginfo("Angle: %s", Angle)
            rospy.loginfo("posterior: %s", posterior)
            rospy.loginfo("most_probable_goal: %s", index+1)
            print('-----------------------------------------------------------------------')
            print("--- %s seconds ---" % (time.time() - start_time))  # show me execution time for each iteration

            intent_pub.publish(index+1)   # publish intent to the topic /most_probable_goal

        rate.sleep()
    except rospy.ROSInterruptException:
        pass
