#!/usr/bin/env python
#from __future__ import division, print_function


# This script estimates the operator's intent (i.e. most probable goal) using Recursive Bayesian Estimation.
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
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import style


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

# compute likelihood
def compute_like_BOIR(path, Angle, wpath, wphi, maxA, maxP):
     a = Angle / maxA
     p = path / maxP
     like = np.exp(-a/wphi) * np.exp(-p/wpath)
     return like

# compute conditional
def compute_cond_BOIR(cond, prior_BOIR):
    sum =  np.matmul(cond, prior_BOIR.T)
    return sum

# compute posterior P(goal|obs) = normalized(likelihood * conditional)
def compute_post_BOIR(likelihood_BOIR, conditional_BOIR):
    out2 = likelihood_BOIR * conditional_BOIR
    post = out2 / np.sum(out2)
    return post


def compute_like_RBII(dis):
     like = np.exp(-5.5*dis)
     return like

# compute conditional
def compute_cond_RBII(cond, prior_RBII):
    sum =  np.matmul(cond, prior_RBII.T)
    return sum

# compute posterior P(goal|obs) = normalized(likelihood * conditional)
def compute_post_RBII(likelihood_RBII, conditional_RBII):
    out2 = likelihood_RBII * conditional_RBII
    post = out2 / np.sum(out2)
    return post

def compute_ECF(dis, term, k):
    Cd = np.exp(-dis)
    C8 = np.exp(((k*term) / 180) - k)
    C = Cd * C8
    C = C / np.sum(C)
    return C

# -------------------------------------------------- F U N C T I O N S --------------------------------------------------------------- #




def run():

    rospy.init_node('bayesian_filter')

    # Create a callable proxy to GetPlan service
    get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)

    # create tf.TransformListener objects
    listener = tf.TransformListener()
    listener1 = tf.TransformListener()
    listener2 = tf.TransformListener()
    listener3 = tf.TransformListener()

    # Subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)

    # Publishers
    pub = rospy.Publisher('most_probable_goal', Float32, queue_size = 1)
    poster1 = rospy.Publisher('poster1', Float32, queue_size = 1)
    poster2 = rospy.Publisher('poster2', Float32, queue_size = 1)
    poster3 = rospy.Publisher('poster3', Float32, queue_size = 1)
    angle1 = rospy.Publisher('angle1', Float32, queue_size = 1)
    angle2 = rospy.Publisher('angle2', Float32, queue_size = 1)
    angle3 = rospy.Publisher('angle3', Float32, queue_size = 1)
    path1 = rospy.Publisher('path1', Float32, queue_size = 1)
    path2 = rospy.Publisher('path2', Float32, queue_size = 1)
    path3 = rospy.Publisher('path3', Float32, queue_size = 1)
    term1 = rospy.Publisher('term1', Float32, queue_size = 1)
    term2 = rospy.Publisher('term2', Float32, queue_size = 1)
    term3 = rospy.Publisher('term3', Float32, queue_size = 1)

    # declare variables for first BAYES
    index = 0
    wphi = 0.6
    wpath = 0.4
    maxA = 180
    maxP = 25
    n = 3   # number of goals
    Delta = 0.2
    k = 10
    x = np.arange(n)

    # Initialize Prior-beliefs according to goals' number
    data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
    prior_BOIR = data0
    prior_RBII = data0

    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data_cpt = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data_cpt, 1-Delta)
    cond = data_cpt


    rate = rospy.Rate(4) # 4 Hz (4 loops/sec) .. (0.25 sec)


    while not rospy.is_shutdown():


        # robot coordinates (MAP FRAME)
        robot_coord = [x_robot, y_robot]
        g1 = [4.60353183746, -13.2964735031] #gleft
        g2 = [6.86671924591, -9.13561820984] #gcenter
        g3 = [4.69921255112, -4.81423997879] #gright

        targets = [g1, g2, g3] # list of FIRST set of goals (MAP FRAME) --> useful for euclidean distance



# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #


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
	rot = yaw_degrees


        # NEW coordinates' goals after transformations (ROBOT FRAME)
        g1_new = [list1.point.x, list1.point.y]
        g2_new = [list2.point.x, list2.point.y]
        g3_new = [list3.point.x, list3.point.y]


        # list of FIRST set of goals (ROBOT FRAME)
        new_goals = [g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]] # list
        new = np.array(new_goals) # array --> useful for angle computation

        # it is needed just for saving the values
        measure = np.array([])
        for z in targets:
            dis = distance.euclidean(robot_coord, z)
            measure = np.append(measure, dis)
        dis = measure


# -------------------------------------------------- T R A N S F O R M A T I O N S --------------------------------------------------------------- #




# 1st OBSERVATION -------------------------------------------------------------------------
        # angles computation between robot (x=0, y=0) & each transformed goal (2nd Observation)
        xs = []
        ys = []
        it = iter(new)
        [(xs.append(i), ys.append(next(it))) for i in it]
        angles = np.arctan2(ys, xs) * 180 / np.pi
        Angle = abs(angles)
        print("Angles: ", Angle)

	term = 180 - abs(rot-Angle)
# 1st OBSERVATION -------------------------------------------------------------------------



# 2nd OBSERVATION -------------------------------------------------------------------------
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
# 2nd OBSERVATION -------------------------------------------------------------------------



# BAYES' FILTER ----------------------------------------------------------------------

        likelihood_BOIR = compute_like_BOIR(path, Angle, wpath, wphi, maxA, maxP)
        conditional_BOIR = compute_cond_BOIR(cond, prior_BOIR)
        posterior_BOIR = compute_post_BOIR(likelihood_BOIR, conditional_BOIR)
        index_BOIR = np.argmax(posterior_BOIR)
        prior_BOIR = posterior_BOIR

        likelihood_RBII = compute_like_RBII(dis)
        conditional_RBII = compute_cond_RBII(cond, prior_RBII)
        posterior_RBII = compute_post_RBII(likelihood_RBII, conditional_RBII)
        index_RBII = np.argmax(posterior_RBII)
        prior_RBII = posterior_RBII

        ecf = compute_ECF(dis, term, k)
        index_ecf = np.argmax(ecf)
        # print(ecf)

# BAYES' FILTER ----------------------------------------------------------------------
        # bars = ('g1', 'g2', 'g3')

        barWidth = 0.2
        r2 = [i + barWidth for i in x]
        r3 = [i + barWidth for i in r2]
        # print(posterior_BOIR)
        # print(posterior_RBII)
        plt.bar(x, posterior_BOIR, width=barWidth, color="aqua", label='BOIR')
        plt.bar(r2, posterior_RBII, width=barWidth, color="steelblue", label='RBII-1')
        plt.bar(r3, ecf, width=barWidth, color="blue", label='ECF')
        plt.title('Probability Mass Function', fontweight='bold')
        plt.xlabel('Goals in Scenario 2', fontweight='bold')
        plt.ylabel('belief/posterior value', fontweight='bold')
        # plt.xticks(x, bars)
        plt.xticks([q + barWidth for q in range(n)], ['Goal1', 'Goal2', 'Goal3'])
        plt.ylim((0, 1))
        plt.yticks(np.arange(0, 1, 0.1))
        # plt.legend()
        plt.legend(loc=2, prop={'size': 13})
        plt.draw()
        plt.pause(0.01)
        plt.clf()


        # print ...
        #rospy.loginfo("rotate: %s", yaw_degrees)
        # rospy.loginfo("len: %s", path)
        # rospy.loginfo("Angles: %s", Angle)
        # rospy.loginfo("Posterior: %s", posterior)
        # rospy.loginfo("Potential Goal is %s", index+1)
        rospy.loginfo("Recognized goal with BOIR : %s", index_BOIR+1)
        rospy.loginfo("Recognized goal with RBII : %s", index_RBII+1)
        rospy.loginfo("Recognized goal with ECF : %s", index_ecf+1)
        print('-------------------------------------------------------')



        # pub.publish(index+1)
        # poster1.publish(posterior[0])
        # poster2.publish(posterior[1])
        # poster3.publish(posterior[2])
        angle1.publish(Angle[0])
        angle2.publish(Angle[1])
        angle3.publish(Angle[2])
        path1.publish(path[0])
        path2.publish(path[1])
        path3.publish(path[2])
	term1.publish(term[0])
	term2.publish(term[1])
	term3.publish(term[2])






        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
