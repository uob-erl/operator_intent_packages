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

# g1 = [18.4734783173, -0.299915790558]
#
# g2 = [22.5236606598, -2.63952469826]
#
#
# g3 = [29.0113964081, -0.833422660828]
#
#
# g4 = [26.8627986908, -6.38541555405]
#
#
# g5 = [23.6445789337, -8.47383022308]
#
#
# g6 =[19.8492908478, -7.68519210815]



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




# FIRST set of goals
G1 = Point()
G2 = Point()
G3 = Point()


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


#compute likelihood : P(Z|goal) = normalized [e^(-k*Z)] , for all goals-distances
def compute_like(path, Angle, wpath, wphi):
    Path_norm = np.array([(path[0] / np.sum(path)), (path[1] / np.sum(path)), (path[2] / np.sum(path))])
    Angle_norm = np.array([(Angle[0] / np.sum(Angle)), (Angle[1] / np.sum(Angle)), (Angle[2] / np.sum(Angle))])
    out0 = (wpath * np.exp(-Path_norm)) * (wphi * np.exp(-Angle_norm))
    like = out0

    return like



def compute_cond(cond, prior):
    #rospy.loginfo("Prior: %s", prior)
    out1 = np.matmul(cond, prior.T)
    sum = out1

    return sum



def compute_decay(n, timing, minimum, check1, check2, check3):
    rospy.loginfo("seconds: %s", timing)
    decay = 0.95 - (0.06 * timing) # window 10 sec means --> NEVER decay under 35% !!!!!
    datakati = np.ones(n-1) * (1-decay)/(n-1)
    updated_prior = datakati
    allo = (1-decay)/(n-1)
    if minimum == check1:
        updated_prior = np.array([decay, allo, allo])
    elif minimum == check2:
        updated_prior = np.array([allo, decay, allo])
    else:
        updated_prior = np.array([allo, allo, decay])

    return updated_prior



def extra_term(summary, dec):
    ex = summary * dec
    extra = ex / np.sum(ex)

    return extra


# compute posterior P(goal|theta) = normalized(likelihood * conditional)
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)

    return post



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


    # Subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)


    # Publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size = 1)
    poster1 = rospy.Publisher('poster1', Float32, queue_size = 1)
    poster2 = rospy.Publisher('poster2', Float32, queue_size = 1)
    poster3 = rospy.Publisher('poster3', Float32, queue_size = 1)


    # declare variables for first BAYES
    posterior = 0
    g_prime_old = 0
    timing = 0
    minimum = 0
    index = 0
    state = 0
    wphi = 0.65
    wpath = 0.35

    l = 0.95

    n = 3   # number of total goals (prime+subgoals)

    Delta = 0.2
    p = (1-l)/(n-1)



    # Initialize Prior-beliefs according to goals' number
    data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
    prior = data0



    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data_cpt = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data_cpt, 1-Delta)
    cond = data_cpt


    rate = rospy.Rate(4) # 4 Hz (4 loops/sec) .. (0.25 sec)


    while not rospy.is_shutdown():


        # robot coordinates (MAP FRAME)
        robot_coord = [x_robot, y_robot]
        g_prime = [x_nav, y_nav]  # CLICKED point - g'
        g1 = [21.9794311523, -11.4826393127] #gleft
        g2 = [21.3006038666, -17.3720340729] #gcenter
        g3 = [4.67607975006, -20.1855487823] #gright



        targets = [g1, g2, g3] # list of FIRST set of goals (MAP FRAME) --> useful for euclidean distance





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





        try:

            (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", Gprime_msg)  # transform g_prime to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y

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




        # NEW robot's coordinates after transformation (we don't care !) --- robot's x,y always 0 in ROBOT FRAME
        # robot = [translation[0], translation[1]]
        # rospy.loginfo("Robot_FRAME: %s", robot)


        # list of FIRST set of goals (ROBOT FRAME)
        new_goals = [g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]] # list
        new = np.array(new_goals) # array --> useful for angle computation

# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #



# -------------------------------------------------- O B S E R V A T I O N S ------------------------------------------------------------- #

        check1 = distance.euclidean(g_prime, g1) #allagi g1 se g_refA
        #rospy.loginfo("Check: %s", check1)

        check2 = distance.euclidean(g_prime, g2) #allagi g1 se g_refA
        #rospy.loginfo("Check: %s", check2)

        check3 = distance.euclidean(g_prime, g3) #allagi g1 se g_refA
        #rospy.loginfo("Check: %s", check3)
        check = [check1, check2, check3]
        minimum = min(check)

        note = any(i<=5 for i in check)
        if note and state == 0 and g_prime != g_prime_old:
            g_prime_old = g_prime
            state = 1
            rospy.loginfo("STATE - BAYES me decay")

            if minimum == check1:
                prior = np.array([l, p, p])
            elif minimum == check2:
                prior = np.array([p, l, p])
            else:
                prior = np.array([p, p, l])


            while (timing < 10):

                rospy.loginfo("STATE - BAYES me decay")
                #rospy.loginfo("likelihood: %s", likelihood)



                wphi = 0.65
                wpath = 0.35

                n = 3  # number of total goals (prime+subgoals)

                Delta = 0.2

                l = 0.95

                rospy.loginfo("prior: %s", prior)

                # creation of Conditional Probability Table 'nxn' according to goals & Delta
                data_cpt = np.ones((n, n)) * (Delta / (n-1))
                np.fill_diagonal(data_cpt, 1-Delta)
                cond = data_cpt


                # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
                robot_base = [0, 0]

                # if n=3 ..
                ind_pos_x = [0, 2, 4]
                ind_pos_y = [1, 3, 5]


                dx = new - robot_base[0]
                dy = new - robot_base[1]
                Dx = dx[ind_pos_x]
                Dy = dx[ind_pos_y]
                angle = np.arctan2(Dy, Dx) * 180 / np.pi
                Angle = abs(angle)


                # generate plan towards goals --> n-path lengths .. (3rd Observation)
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
                path = length

                #rospy.loginfo("Prior would be: %s", prior)
                likelihood = compute_like(path, Angle, wpath, wphi)
                dec = compute_decay(n, timing, minimum, check1, check2, check3)
                summary = compute_cond(cond, prior)
                posterior = compute_post(likelihood, summary)
                plus = extra_term(summary, dec)
                final = compute_final(likelihood, plus)
                index = np.argmax(final)
                prior = final

                # print ...
                #rospy.loginfo("rotate: %s", yaw_degrees)
                #rospy.loginfo("len: %s", path)
                #rospy.loginfo("Angles: %s", Angle)
                rospy.loginfo("decay: %s", dec)
                rospy.loginfo("Posterior would be: %s", posterior)
                rospy.loginfo("Posterior: %s", final)
                rospy.loginfo("Potential Goal is %s", index+1)

                timing = timing + 1
                rospy.sleep(0.5)


                pub.publish(index+1)



        else:
            state = 0

            rospy.loginfo("STATE - BAYES normal")
            wphi = 0.65
            wpath = 0.35

            n = 3   # number of total goals (prime+subgoals)

            Delta = 0.2

            # Initialize Prior-beliefs according to goals' number
            # data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
            # prior = data0
            #prior = posterior

            rospy.loginfo("Prior: %s", prior)

            # creation of Conditional Probability Table 'nxn' according to goals & Delta
            data_cpt = np.ones((n, n)) * (Delta / (n-1))
            np.fill_diagonal(data_cpt, 1-Delta)
            cond = data_cpt


            # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
            robot_base = [0, 0]

            # if n=3 ..
            ind_pos_x = [0, 2, 4]
            ind_pos_y = [1, 3, 5]


            dx = new - robot_base[0]
            dy = new - robot_base[1]
            Dx = dx[ind_pos_x]
            Dy = dx[ind_pos_y]
            angle = np.arctan2(Dy, Dx) * 180 / np.pi
            Angle = abs(angle)


            # generate plan towards goals --> n-path lengths .. (3rd Observation)
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
                rospy.sleep(0.05) # 0.05 x 4 = 0.2 sec

                length = np.append(length, path_length)
            path = length

            #rospy.loginfo("Prior: %s", prior)
            likelihood = compute_like(path, Angle, wpath, wphi)
            conditional = compute_cond(cond, prior)
            posterior = compute_post(likelihood, conditional)
            index = np.argmax(posterior)
            prior = posterior

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



        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
