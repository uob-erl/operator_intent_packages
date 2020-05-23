#!/usr/bin/env python
#from __future__ import division, print_function


# main filter -- bayesian estimation with 2 observations ( 3 in total --> 2 after combining)
# OBS : euclidean distance -- angle -- path length
# Simulated environment : 1)husky_gazebo_intent.launch
#                         2)husky_gazebo_mi_experiment.launch

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
from std_msgs.msg import Int32, Float32, Int8
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from random import randint


# 2)husky_gazebo_mi_experiment.launch
# g1 = [20.9835529327, 11.3141422272]
# g2 = [33.935836792, 9.45084857941]
# g3 = [13.5039768219, -10.5052185059]
# g4 = [12.3510360718, -16.0753192902]


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
path_length = 0
checkA = 0
checkB = 0
state = 0




Gprime = Point() # operator's nav goal

# FIRST set of goals
G1 = Point()
G2 = Point()

# SECOND set of goals
G3 = Point()
G4 = Point()

Goal = PoseStamped()
Start = PoseStamped()





# -------------------------------------------------- C A L L B A C K S --------------------------------------------------------------- #

# callback function for navigational goal (g')
def call_nav(msg):
    global x_nav, y_nav, qx_nav, qy_nav, qz_nav, qw_nav
    x_nav = msg.pose.position.x
    y_nav = msg.pose.position.y
    qx_nav = msg.pose.orientation.x
    qy_nav = msg.pose.orientation.y
    qz_nav = msg.pose.orientation.z
    qw_nav = msg.pose.orientation.w


# callback function for robot's coordinates (MAP FRAME)
def call_rot(message):
    global x_robot, y_robot, qx, qy, qz, qw
    x_robot = message.pose.pose.position.x
    y_robot = message.pose.pose.position.y
    qx = message.pose.pose.orientation.x
    qy = message.pose.pose.orientation.y
    qz = message.pose.pose.orientation.z
    qw = message.pose.pose.orientation.w


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
def compute_like(path, Angle, wpath, wphi):
    Path_norm = np.array([(path[0] / np.sum(path)), (path[1] / np.sum(path)), (path[2] / np.sum(path))])
    Angle_norm = np.array([(Angle[0] / np.sum(Angle)), (Angle[1] / np.sum(Angle)), (Angle[2] / np.sum(Angle))])
    #print(np.sum(Angle))
    out0 = (wpath * np.exp(-Path_norm)) * (wphi * np.exp(-Angle_norm))
    #out0 = np.exp(-wpath * Path_norm) * np.exp(-wphi * Angle_norm)
    like = out0
    return like

# compute conditional : normalized [Sum P(goal.t|goal.t-1) * b(goal.t-1)] .. "ACTION model"
def compute_cond(cond, prior):
    out1 =  np.matmul(cond, prior.T)
    sum = out1 #/ np.sum(out1)
    return sum


# compute posterior P(goal|obs) = normalized(likelihood * conditional)
def compute_post(likelihood, conditional):
    out2 = likelihood * conditional
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
    listener1 = tf.TransformListener()
    listener2 = tf.TransformListener()
    listener3 = tf.TransformListener()
    listener4 = tf.TransformListener()



    # Subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)


    # Publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size = 1)
    output = rospy.Publisher('operator_intent', Int8, queue_size = 1)
    area = rospy.Publisher('current_area', Int8, queue_size = 1)
    #stoxos = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=5)


    # declare variables for first BAYES
    current = 0
    index = 0
    wphi = 0.75
    wpath = 0.25

    n = 3   # number of total goals (prime+subgoals)
    l = 0.7
    Delta = 0.2


    # Initialize Prior-beliefs according to goals' number .. nav goal should have higher prior belief at t=0
    data_prior = np.ones(n-1) * (1-l)/(n-1)   # P(g_prime)=0.75 , P(g1)= ... , P(g2)= ...
    prior = data_prior
    prior = np.insert(prior, 0, l, axis=0)


    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data_cpt = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data_cpt, 1-Delta)
    cond = data_cpt


    rate = rospy.Rate(4) # 4 Hz (4 loops/sec) .. (0.25 sec)


    while not rospy.is_shutdown():


        # robot coordinates (MAP FRAME)
        robot_coord = [x_robot, y_robot]

        # goals coordinates (MAP FRAME)
        g_prime = [x_nav, y_nav]  # CLICKED point - g'

        g_refA = [26.4987888336, 7.16181945801]
        g_refB = [17.3956737518, -12.9359521866]


        # g1 = [20.983089447021484, 12.749549865722656]
        # g2 = [33.75846862792969, 10.814520835876465]
        g1 = [20.9167137146, 12.6046304703]
        g2 = [33.3691520691, 11.5497436523]



        g3 = [12.8780117035, -10.7926425934]
        g4 = [14.0271482468, -17.25157547]






        targets_A = [g_prime, g1, g2] # list of FIRST set of goals (MAP FRAME) --> useful for euclidean distance
        targets_B = [g_prime, g3, g4] # list of SECOND set of goals (MAP FRAME) --> useful for euclidean distance

        targets_A_2 = [g1, g2]
        targets_B_2 = [g3, g4]



        # # FIRST set with offset --> useful for GetPlan
        #
        # g1_off = [20.983089447021484, 12.749549865722656]
        # g2_off = [33.75846862792969, 10.814520835876465]
        # # g1_off = [21.0046730042, 11.751698494]
        # # g2_off = [33.9431762695, 9.90261650085]
        #
        # targets_A_off = [g_prime, g1_off, g2_off]
        # targets_A_off_2 = [g1_off, g2_off]
        #
        #
        # # SECOND set with offset --> useful for GetPlan
        # g3_off = [13.0466022491, -10.525103569]
        # g4_off = [12.5871200562, -16.5315208435]
        #
        # targets_B_off = [g_prime, g3_off, g4_off]
        # targets_B_off_2 = [g3_off, g4_off]



# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #

        # prepare transformation from g_prime(MAP FRAME) to gprime -> g_prime_new(ROBOT FRAME)
        Gprime_msg = PointStamped()
        Gprime_msg.header.frame_id = "map"
        Gprime_msg.header.stamp = rospy.Time(0)
        Gprime_msg.point.x = g_prime[0]
        Gprime_msg.point.y = g_prime[1]

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

        # prepare transformation from g5(MAP FRAME) to g5 -> g5_new(ROBOT FRAME)
        G4_msg = PointStamped()
        G4_msg.header.frame_id = "map"
        G4_msg.header.stamp = rospy.Time(0)
        G4_msg.point.x = g4[0]
        G4_msg.point.y = g4[1]





        try:

            (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", Gprime_msg)  # transform g_prime to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y
            list4 = listener4.transformPoint("/base_link", G4_msg)  # transform g4 to base_link (ROBOT FRAME) , returns x,y

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):

            continue


        # convert from quaternion to RPY (the yaw of the robot in /map frame)
        orientation_list = rotation

        # rotation angle of the robot = yaw in ROBOT FRAME
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        yaw_degrees = yaw * 180 / np.pi


        # NEW coordinates' goals after transformations (ROBOT FRAME)
        g_prime_new = [list_nav.point.x, list_nav.point.y]
        g1_new = [list1.point.x, list1.point.y]
        g2_new = [list2.point.x, list2.point.y]
        g3_new = [list3.point.x, list3.point.y]
        g4_new = [list4.point.x, list4.point.y]



        # NEW robot's coordinates after transformation (we don't care !) --- robot's x,y always 0 in ROBOT FRAME
        # robot = [translation[0], translation[1]]
        # rospy.loginfo("Robot_FRAME: %s", robot)


        # list of FIRST set of goals (ROBOT FRAME)
        new_goals_A = [g_prime_new[0], g_prime_new[1], g1_new[0], g1_new[1], g2_new[0], g2_new[1]] # list
        new_A = np.array(new_goals_A) # array --> useful for angle computation

        new_goals_A_2 = [g1_new[0], g1_new[1], g2_new[0], g2_new[1]] # list
        new_A_2 = np.array(new_goals_A_2)


        # list of SECOND set of goals (ROBOT FRAME)
        new_goals_B = [g_prime_new[0], g_prime_new[1], g3_new[0], g3_new[1], g4_new[0], g4_new[1]] # list
        new_B = np.array(new_goals_B) # array --> useful for angle computation

        new_goals_B_2 = [g3_new[0], g3_new[1], g4_new[0], g4_new[1]] # list
        new_B_2 = np.array(new_goals_B_2)



# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #



# -------------------------------------------------- O B S E R V A T I O N S ------------------------------------------------------------- #

        # check euclidean distance(nav-g1) to define the proper set of goals (either 1,2,3 OR 5,6,7)
        # WE CAN ADD MORE HERE !!!
        checkA = distance.euclidean(g_prime, g_refA) #allagi g1 se g_refA
        rospy.loginfo("CheckA: %s", checkA)

        checkB = distance.euclidean(g_prime, g_refB)
        rospy.loginfo("CheckB: %s", checkB)


        if checkA < 8 and checkB > 15: # area A

            state = 1 # area A = area 1
            rospy.loginfo("area A")


            rospy.loginfo("status = prwto") # EXPLORE OFF

            # decision = 0
            # w1 = 0.05
            # w2 = 0.75
            # w3 = 0.2
            # n = 3   # number of total goals (prime+subgoals)
            # l = 0.7
            # Delta = 0.2


            # # Initialize Prior-beliefs according to goals' number .. nav goal should have higher prior belief at t=0
            # data_prior = np.ones(n-1) * (1-l)/(n-1)   # P(g_prime)=0.75 , P(g1)= ... , P(g2)= ...
            # prior = data_prior
            # prior = np.insert(prior, 0, l, axis=0)
            #
            #
            # # creation of Conditional Probability Table 'nxn' according to goals & Delta
            # data_cpt = np.ones((n, n)) * (Delta / (n-1))
            # np.fill_diagonal(data_cpt, 1-Delta)
            # cond = data_cpt

            # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
            measure = np.array([])
            for x in targets_A:
                dist = distance.euclidean(robot_coord, x)
                measure = np.append(measure, dist)
            dist = measure
            rospy.loginfo("Distance: %s", dist)


            # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
            robot_base = [0, 0]

            # if n=3 ..
            ind_pos_x = [0, 2, 4]
            ind_pos_y = [1, 3, 5]


            dx = new_A - robot_base[0]
            dy = new_A - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)


            # generate plan towards goals --> n-path lengths .. (3rd Observation)
            length = np.array([])
            for j in targets_A:

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
            path = length


            # BAYES' FILTER
            likelihood = compute_like(path, Angle, wpath, wphi)
            conditional = compute_cond(cond, prior)
            posterior = compute_post(likelihood, conditional)
            index = np.argmax(posterior)
            prior = posterior

            # print ...
            rospy.loginfo("rotate: %s", yaw_degrees)
            rospy.loginfo("len: %s", path)
            rospy.loginfo("OPERATOR_goal: %s", g_prime)
            rospy.loginfo("Angles: %s", Angle)
            rospy.loginfo("Posterior: %s", posterior)
            rospy.loginfo("Potential Goal is %s", index)







        #else:   # area B

        elif checkA > 15 and checkB < 8:   # area B

            # edw to allo if opws panw
            state = 2 # area B = area 2
            rospy.loginfo("area B")


            rospy.loginfo("status = prwto") # EXPLORE OFF

            # decision = 0
            # w1 = 0.05
            # w2 = 0.75
            # w3 = 0.2
            # n = 3   # number of total goals (prime+subgoals)
            # l = 0.7
            # Delta = 0.2


            # # Initialize Prior-beliefs according to goals' number .. nav goal should have higher prior belief at t=0
            # data_prior = np.ones(n-1) * (1-l)/(n-1)   # P(g_prime)=0.75 , P(g1)= ... , P(g2)= ...
            # prior = data_prior
            # prior = np.insert(prior, 0, l, axis=0)
            #
            #
            # # creation of Conditional Probability Table 'nxn' according to goals & Delta
            # data_cpt = np.ones((n, n)) * (Delta / (n-1))
            # np.fill_diagonal(data_cpt, 1-Delta)
            # cond = data_cpt

            # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
            measure = np.array([])
            for x in targets_B:
                dist = distance.euclidean(robot_coord, x)
                measure = np.append(measure, dist)
            dist = measure
            rospy.loginfo("Distance: %s", dist)


            # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
            robot_base = [0, 0]

            # if n=3 ..
            ind_pos_x = [0, 2, 4]
            ind_pos_y = [1, 3, 5]


            dx = new_B - robot_base[0]
            dy = new_B - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)


            # generate plan towards goals --> n-path lengths .. (3rd Observation)
            length = np.array([])
            for j in targets_B:

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
            path = length


            # BAYES' FILTER
            likelihood = compute_like(path, Angle, wpath, wphi)
            conditional = compute_cond(cond, prior)
            posterior = compute_post(likelihood, conditional)
            index = np.argmax(posterior)
            prior = posterior

            # print ...
            rospy.loginfo("rotate: %s", yaw_degrees)
            rospy.loginfo("len: %s", path)
            rospy.loginfo("OPERATOR_goal: %s", g_prime)
            rospy.loginfo("Angles: %s", Angle)
            rospy.loginfo("Posterior: %s", posterior)
            rospy.loginfo("Potential Goal is %s", index)




        elif checkA + checkB > 25: # area C - end
            state = 3 # area C = area 3
            rospy.loginfo("moving towards END") # EXPLORE OFF
            decision = 0




        pub.publish(index)
        area.publish(state)
        #output.publish(decision)


        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
