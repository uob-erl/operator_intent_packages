#!/usr/bin/env python
#from __future__ import division, print_function

import rospy
import numpy as np
import random
import math
import tf
import time
import actionlib
from actionlib import ActionServer
from actionlib_msgs.msg import GoalStatus
from actionlib_msgs.msg import GoalStatus, GoalID
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
check1 = 0
check2 = 0
check3 = 0
check4 = 0
check5 = 0
check6 = 0
check7 = 0
state = 0


# set of goals
G1 = Point()
G2 = Point()
G3 = Point()
G4 = Point()
G5 = Point()
G6 = Point()
G7 = Point()

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


# compute Likelihood : P(Z|goal) = W1*exp(-angle) * W2*exp(-path), for all goals
def compute_like(path, Angle, wpath, wphi):
    Path_norm = np.array([(path[0]/np.sum(path)), (path[1]/np.sum(path)), (path[2]/np.sum(path)), (path[3]/np.sum(path)), (path[4]/np.sum(path)), (path[5]/np.sum(path)), (path[6]/np.sum(path))])
    Angle_norm = np.array([(Angle[0]/np.sum(Angle)), (Angle[1]/np.sum(Angle)), (Angle[2]/np.sum(Angle)), (Angle[3]/np.sum(Angle)), (Angle[4]/np.sum(Angle)), (Angle[5]/np.sum(Angle)), (Angle[6]/np.sum(Angle))])
    out0 = (wpath * np.exp(-Path_norm)) * (wphi * np.exp(-Angle_norm))
    like = out0
    return like



# compute transition model
def compute_cond(cond, prior):
    out1 = np.matmul(cond, prior.T)
    sum = out1
    return sum



# compute posterior P(goal|Z) = normalized(Likelihood * transition)  # Normal BAYES
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)
    return post



# given desired initial-clicked probability=P0, time-window, and final-clicked probability=threshold
# compute the parameters (slope, interY) of the appropriate linear decay function
def equation(P0, window, threshold):
    deltaX = window - 0
    deltaY = P0 - threshold
    slope = deltaY / deltaX
    interY = P0
    return slope, interY



# compute the linear decay function given the parameters ...
# 1st loop (t=0) --> decay[] = prior[]
# 2nd loop --> decay[] != prior
# .....
# prior[] takes the posterior's values AND decay[] takes normalized reduced values based on equation
def compute_decay(n, timing, minimum, check1, check2, check3, check4, check5, check6, check7, slope, interY):
    rospy.loginfo("seconds: %s", timing)
    decay = interY - (slope * timing) # window = 10sec here means --> NEVER decay under 35%  = threshold !!!!!
    datadec = np.ones(n-1) * (1-decay)/(n-1)
    updated = datadec
    rest = (1-decay)/(n-1)
    if minimum == check1:
        updated = np.array([decay, rest, rest, rest, rest, rest, rest])
    elif minimum == check2:
        updated = np.array([rest, decay, rest, rest, rest, rest, rest])
    elif minimum == check3:
        updated = np.array([rest, rest, decay, rest, rest, rest, rest])
    elif minimum == check4:
        updated = np.array([rest, rest, rest, decay, rest, rest, rest])
    elif minimum == check5:
        updated = np.array([rest, rest, rest, rest, decay, rest, rest])
    elif minimum == check6:
        updated = np.array([rest, rest, rest, rest, rest, decay, rest])
    else:
        updated = np.array([rest, rest, rest, rest, rest, rest, decay])

    return updated


# compute transition model   # BAYES with decay
def extra_term(summary, dec):
    ex = summary * dec
    extra = ex #/ np.sum(ex)
    return extra


# compute posterior   # BAYES with decay
def compute_final(likelihood, plus):
    out = likelihood * plus
    poster = out / np.sum(out)
    return poster

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
    listener5 = tf.TransformListener()
    listener6 = tf.TransformListener()
    listener7 = tf.TransformListener()


    # Subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)


    # Publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size = 1)
    poster1 = rospy.Publisher('poster1', Float32, queue_size = 1)
    poster2 = rospy.Publisher('poster2', Float32, queue_size = 1)
    poster3 = rospy.Publisher('poster3', Float32, queue_size = 1)
    poster4 = rospy.Publisher('poster4', Float32, queue_size = 1)
    poster5 = rospy.Publisher('poster5', Float32, queue_size = 1)
    poster6 = rospy.Publisher('poster6', Float32, queue_size = 1)
    poster7 = rospy.Publisher('poster7', Float32, queue_size = 1)


    # initializations
    P0 = 0.95
    window = 10
    threshold = 0.35
    g_prime_old = 0
    timing = 0
    minimum = 0
    value = 4
    index = 0
    state = 0
    wphi = 0.65  # angle weight
    wpath = 0.35  # path weight
    n = 7   # number of goals
    Delta = 0.2
    p = (1-P0)/(n-1) # rest of prior values in decay mode



    # Initialize Prior-beliefs according to goals' number
    data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
    prior = data0



    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data_cpt = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data_cpt, 1-Delta)
    cond = data_cpt


    rate = rospy.Rate(2) # 4 Hz (4 loops/sec) .. (0.5 sec)


    while not rospy.is_shutdown():


        # robot coordinates (MAP FRAME)
        robot_coord = [x_robot, y_robot]
        g_prime = [x_nav, y_nav]  # CLICKED point - g'
        g1 = [13.384099, -0.070828]
        g2 = [18.697673, -0.090648]
        g3 = [23.227645, -1.249789]
        g4 = [29.208362, -0.728264]
        g5 = [27.863958, -6.041914]
        g6 = [23.862051, -8.185440]
        g7 = [20.085504, -7.524883]
        targets = [g1, g2, g3, g4, g5, g6, g7] # list of FIRST set of goals (MAP FRAME) --> useful for euclidean distance



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

        G4_msg = PointStamped()
        G4_msg.header.frame_id = "map"
        G4_msg.header.stamp = rospy.Time(0)
        G4_msg.point.x = g4[0]
        G4_msg.point.y = g4[1]

        G5_msg = PointStamped()
        G5_msg.header.frame_id = "map"
        G5_msg.header.stamp = rospy.Time(0)
        G5_msg.point.x = g5[0]
        G5_msg.point.y = g5[1]

        G6_msg = PointStamped()
        G6_msg.header.frame_id = "map"
        G6_msg.header.stamp = rospy.Time(0)
        G6_msg.point.x = g6[0]
        G6_msg.point.y = g6[1]

        G7_msg = PointStamped()
        G7_msg.header.frame_id = "map"
        G7_msg.header.stamp = rospy.Time(0)
        G7_msg.point.x = g7[0]
        G7_msg.point.y = g7[1]



        try:

            (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", Gprime_msg)  # transform g_prime to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y
            list4 = listener4.transformPoint("/base_link", G4_msg)
            list5 = listener5.transformPoint("/base_link", G5_msg)
            list6 = listener6.transformPoint("/base_link", G6_msg)
            list7 = listener7.transformPoint("/base_link", G7_msg)

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
        g5_new = [list5.point.x, list5.point.y]
        g6_new = [list6.point.x, list6.point.y]
        g7_new = [list7.point.x, list7.point.y]



        # list of FIRST set of goals (ROBOT FRAME)
        new_goals = [g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1], g4_new[0], g4_new[1], g5_new[0], g5_new[1], g6_new[0], g6_new[1], g7_new[0], g7_new[1]] # list
        new = np.array(new_goals) # array --> useful for angle computation

# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #



        # ! given the clicked point decide the potential goal !
        # Simply, e.g. if i choose to click around g2, then the euclidean between g2 and click is minimum
        # so g2 prior becomes greater than others ---->  decay function starts to be computed ---> BAYES with Decay
        check1 = distance.euclidean(g_prime, g1)
        #rospy.loginfo("Check1: %s", check1)

        check2 = distance.euclidean(g_prime, g2)
        #rospy.loginfo("Check2: %s", check2)

        check3 = distance.euclidean(g_prime, g3)
        #rospy.loginfo("Check3: %s", check3)

        check4 = distance.euclidean(g_prime, g4)
        check5 = distance.euclidean(g_prime, g5)
        check6 = distance.euclidean(g_prime, g6)
        check7 = distance.euclidean(g_prime, g7)

        check = [check1, check2, check3, check4, check5, check6, check7]
        minimum = min(check)

        note = any(i<=value for i in check)
        if note and state == 0 and g_prime != g_prime_old:
            g_prime_old = g_prime
            state = 1
            rospy.loginfo("STATE - BAYES me decay")

            if minimum == check1:
                prior = np.array([P0, p, p, p, p, p, p])
            elif minimum == check2:
                prior = np.array([p, P0, p, p, p, p, p])
            elif minimum == check3:
                prior = np.array([p, p, P0, p, p, p, p])
            elif minimum == check4:
                prior = np.array([p, p, p, P0, p, p, p])
            elif minimum == check5:
                prior = np.array([p, p, p, p, P0, p, p])
            elif minimum == check6:
                prior = np.array([p, p, p, p, p, P0, p])
            else:
                prior = np.array([p, p, p, p, p, p, P0])


            while (timing < 10):
                # decay function starts to be computed

                rospy.loginfo("STATE - BAYES me decay")

                rospy.loginfo("prior: %s", prior)



    # 1st OBSERVATION -------------------------------------------------------------------------
                # angles computation between robot (x=0, y=0) & each transformed goal (1st Observation)
                robot_base = [0, 0]

                # if n=7 ..
                ind_pos_x = [0, 2, 4, 6, 8, 10, 12]
                ind_pos_y = [1, 3, 5, 7, 9, 11, 13]


                dx = new - robot_base[0]
                dy = new - robot_base[1]
                Dx = dx[ind_pos_x]
                Dy = dx[ind_pos_y]
                angle = np.arctan2(Dy, Dx) * 180 / np.pi
                Angle = abs(angle)
    # 1st OBSERVATION -------------------------------------------------------------------------



    # 2nd OBSERVATION ---------------------------------------------------------------------------
                # generate plan towards goals --> n-path lengths .. (2nd Observation)
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
                    rospy.sleep(0.05) # 0.05 x 7 = 0.35 sec

                    length = np.append(length, path_length)
                path = length
    # 2nd OBSERVATION ---------------------------------------------------------------------------





    # BAYES' FILTER with Decay ------------------------------------------------

                # compute parameter's decay equation
                [slope, interY] = equation(P0, window, threshold)
                rospy.loginfo("slope: %s", slope)
                rospy.loginfo("interY %s", interY)

                likelihood = compute_like(path, Angle, wpath, wphi)
                dec = compute_decay(n, timing, minimum, check1, check2, check3, check4, check5, check6, check7, slope, interY)

                summary = compute_cond(cond, prior)

                # what posterior trully is with extra term
                plus = extra_term(summary, dec)
                posterior = compute_final(likelihood, plus)

                index = np.argmax(posterior)
                prior = posterior

    # BAYES' FILTER with Decay------------------------------------------------


                # print ...
                #rospy.loginfo("rotate: %s", yaw_degrees)
                #rospy.loginfo("len: %s", path)
                #rospy.loginfo("Angles: %s", Angle)
                rospy.loginfo("decay: %s", dec)
                rospy.loginfo("POSTERIOR: %s", posterior)
                rospy.loginfo("Potential Goal is %s", index+1)

                timing = timing + 1
                rospy.sleep(0.5)


                pub.publish(index+1)



        else:

            timing = 0
            state = 0

            rospy.loginfo("STATE - BAYES normal")

            rospy.loginfo("Prior: %s", prior)



    # 1st OBSERVATION ---------------------------------------------------------------------------
            # angles computation between robot (x=0, y=0) & each transformed goal (1st Observation)
            robot_base = [0, 0]

            # if n=7 ..
            ind_pos_x = [0, 2, 4, 6, 8, 10, 12]
            ind_pos_y = [1, 3, 5, 7, 9, 11, 13]


            dx = new - robot_base[0]
            dy = new - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)
    # 1st OBSERVATION ---------------------------------------------------------------------------


    # 2nd OBSERVATION ---------------------------------------------------------------------------
            # generate plan towards goals --> n-path lengths .. (2nd Observation)
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
                rospy.sleep(0.05) # 0.05 x 7 = 0.35 sec

                length = np.append(length, path_length)
            path = length
    # 2nd OBSERVATION ---------------------------------------------------------------------------


    # BAYES' FILTER ------------------------------------------------

            likelihood = compute_like(path, Angle, wpath, wphi)
            conditional = compute_cond(cond, prior)
            posterior = compute_post(likelihood, conditional)
            index = np.argmax(posterior)
            prior = posterior

    # BAYES' FILTER ------------------------------------------------


            # print ...
            #rospy.loginfo("rotate: %s", yaw_degrees)
            #rospy.loginfo("len: %s", path)
            #rospy.loginfo("Angles: %s", Angle)

            rospy.loginfo("Posterior: %s", posterior)
            rospy.loginfo("Potential Goal is %s", index+1)




            pub.publish(index+1)
            poster1.publish(posterior[0])
            poster2.publish(posterior[1])
            poster3.publish(posterior[2])
            poster4.publish(posterior[3])
            poster5.publish(posterior[4])
            poster6.publish(posterior[5])
            poster7.publish(posterior[6])



        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
