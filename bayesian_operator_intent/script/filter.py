#!/usr/bin/env python
#from __future__ import division, print_function

import rospy
import numpy as np
import random
import math

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial import distance
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from random import randint
# from filterpy.discrete_bayes import normalize
# from filterpy.discrete_bayes import update



# G0
# position:
#       x: -2.83128278901
#       y: -3.62014215797
#
# G1
# position:
#       x: -0.215494300143
#       y: -5.70071558441
#
# G2
# position:
#       x: 3.08670737031
#       y: -5.93227324436
#
# G3
# position:
#       x: 6.94527384787
#       y: -11.717640655
#
# G4
# position:
#       x: 8.51079199583
#       y: 4.698199058
#
# G5
# position:
#       x: 9.00735322774
#       y: -0.677194482712
#
# G6
# position:
#       x: 11.8716085357
#       y: -0.506166845716
#
# G7
# position:
#       x: 14.9899956087
#       y: 5.20262059311
#
# G8
# position:
#       x: 20.101826819
#       y: -8.62395824827
#
# G9
# position:
#       x: 22.9649931871
#       y: 10.4371948525
#
# G10
# position:
#       x: 25.5818735158
#       y: 15.74648043
#
# G11
# position:
#       x: 27.8554619121
#       y: 6.78231736479



x0 = 0.0
y0 = 0.0
rot_angle = 0
qx = 0
qy = 0
qz = 0
qw = 0
roll = 0
pitch = 0
yaw = 0


# def fnc_callback(msg):
#     # dinw egw times stis coordinates tou robot
#     global x0, y0
#     x0 = msg.linear.xPoseWithCovarianceStamped
#     y0 = msg.linear.y

def call_rot(mes):
    global x0, y0, qx, qy, qz, qw
    x0 = mes.pose.pose.position.x
    y0 = mes.pose.pose.position.y
    qx = mes.pose.pose.orientation.x
    qy = mes.pose.pose.orientation.y
    qz = mes.pose.pose.orientation.z
    qw = mes.pose.pose.orientation.w



# def call(message):
#     global rot_angle
#     rot_angle = message.data



# def callback(msg):
#     global x0, y0
#     x0 = msg.pose.pose.position.x
#     y0 = msg.pose.pose.position.y


# compute likelihood : P(theta|goal) = normalized [e^(-k*theta)] , for all goals-distances
def compute_like(theta, k):
    out0 = np.exp(-k * theta)
    like = out0 / np.sum(out0)

    return like


# compute conditional : normalized [SP(goal.t|goal.t-1) * b(goal.t-1)]
def compute_conditional(cond, prior):
    out1 = np.matmul(cond, prior.T)
    sum = out1 / np.sum(out1)

    return sum


# compute posterior P(goal|theta) = normalized(likelihood * conditional)
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)

    return post



def run():

    rospy.init_node('NODE_SUB_AND_PUB')

    #rot_sub = rospy.Subscriber('/kati', Float32, call) #prepei na allaksw to /kati me ena ros topic pou mou dinei stoixeia gia tin rot angle
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    #rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    # sub=rospy.Subscriber('rand_no', Twist, fnc_callback)
    pub=rospy.Publisher('sub_pub', Float32, queue_size=1)


    k = 2.5
    n = 3  # number of goals
    Delta = 0.2

    # Initialize Prior-beliefs according to goals' number
    data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
    prior = data0

    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data1 = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data1, 1-Delta)
    cond = data1


    rate=rospy.Rate(10)

    while not rospy.is_shutdown():

        robot_coord = [x0, y0]
        g0 = [-2.83128278901, -3.62014215797]
        g1 = [-0.215494300143, -5.70071558441]
        g2 = [3.08670737031, -5.93227324436]
        targets = [g0, g1, g2]
        #goals_coord = [[-2.83128278901, -3.62014215797], [-0.215494300143, -5.70071558441], [3.08670737031, -5.93227324436]] # tha einai PANTA etoimoi


        orientation_list = [qx, qy, qz, qw]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        #quat = quaternion_from_euler(roll, pitch, yaw)
        yaw_degrees = yaw * 180 / np.pi
        rot_angle = yaw_degrees


        psi = np.radians(rot_angle)
        c = np.cos(psi)
        s = np.sin(psi)
        R = np.array(((c, -s), (s, c)))
        
        Goals = np.array([])
        for j in targets:
            robot_new = np.matmul(R, robot_coord)
            ending = np.matmul(R, j)
            Goals = np.append(Goals, ending)
        goals = Goals


        measure = np.array([])
        for p in targets:
            dist = distance.euclidean(robot_coord, p)
            measure = np.append(measure, dist)
        theta = measure



        ind_pos_x = [0, 2, 4]
        ind_pos_y = [1, 3, 5]

        dx = goals - robot_new[0]
        dy = goals - robot_new[1]
        Dx = dx[ind_pos_x]
        Dy = dx[ind_pos_y]

        angle_f = np.array([])
        angle = (np.arctan2(Dy, Dx) * 180 / np.pi)
        for v in angle:
            if (v < 0.0):
                v += 360.0
            else:
                v = v
            angle_f = np.append(angle_f, v)

        likelihood = compute_like(theta, k)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        index = np.argmax(posterior)
        prior = posterior

        rospy.loginfo("ARXIKA XY: %s", robot_coord)
        rospy.loginfo("TELIKA XY: %s", robot_new)
        rospy.loginfo("Targets: %s", targets)
        rospy.loginfo("Goals: %s", goals)
        # rospy.loginfo("Posterior: %s", posterior)
        # rospy.loginfo("Potential Goal is %s", index)

        rospy.loginfo("Theta: %s", theta)
        rospy.loginfo("PHI apo goal: %s", angle)
        rospy.loginfo("PHI apo goal: %s", angle_f)
        rospy.loginfo("Rotation Angle: %s", psi*180/np.pi)
        #rospy.loginfo("QUAT: %s", quat)
        rospy.loginfo("YAW: %s", yaw_degrees)


        pub.publish(index)


        rate.sleep()


if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
