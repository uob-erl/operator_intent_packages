#!/usr/bin/env python
from __future__ import division, print_function

import rospy
import numpy as np
import random
import math

from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from random import randint
from filterpy.discrete_bayes import normalize
from filterpy.discrete_bayes import update


# d1 = 2.0
# def fnc_callback(msg):
#     global d1
#     d1 = msg.data
#     rospy.loginfo(d1)

d1 = 2.0
d2 = 2.5
d3 = 1.2
theta = np.array([d1, d2, d3])
def fnc_callback(msg):
    global d1, d2, d3
    d1 = msg.linear.x
    d2 = msg.linear.y
    d3 = msg.linear.z
    rospy.loginfo(theta)


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

    sub=rospy.Subscriber('rand_no', Twist, fnc_callback)
    pub=rospy.Publisher('sub_pub', Float32, queue_size=1)

    #d1 = 2.0  # sxolio
    #d2 = 2.5
    #d3 = 1.2

    k = 2.5
    n = 3  # goals
    Delta = 0.2

    # Initialize Prior-beliefs according to goals' number
    data0 = np.ones(n) * 1/n   # P(g1)=0.33 , P(g2)=0.33, P(g3)=0.33
    prior = data0

    # creation of Conditional Probability Table 'nxn' according to goals n & Delta
    data1 = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data1, 1-Delta)
    cond = data1


    rate=rospy.Rate(10)

    while not rospy.is_shutdown():

        theta = np.array([d1, d2, d3])  # uparxei


        likelihood = compute_like(theta, k)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        index = np.argmax(posterior)
        prior = posterior


        #rospy.loginfo(likelihood)
        rospy.loginfo(posterior)
        rospy.loginfo(index)
        #rospy.loginfo(d1)
        rospy.loginfo(theta)




        pub.publish(index)

        rate.sleep()


if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
