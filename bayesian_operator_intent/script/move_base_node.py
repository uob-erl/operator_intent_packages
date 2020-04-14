#!/usr/bin/env python
#from __future__ import division, print_function


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




loa = 0
index = 0


def loa_callback(msg):
    global loa
    loa = msg.data

def index_callback(message):
    global index
    index = message.data


def run():

    rospy.init_node('MoveBaseNode')

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    #sub_loa = rospy.Subscriber("/loa", Int8, loa_callback)
    sub_index = rospy.Subscriber('/possible_goal', Float32, index_callback)





    rate = rospy.Rate(5)



    while not rospy.is_shutdown():

        exit = [loa, index]
        #rospy.loginfo("loa: %s", exit[0])
        rospy.loginfo("index: %s", exit[1])

        g1_off = [21.0046730042, 11.751698494]
        g2_off = [33.9431762695, 9.90261650085]


        g3_off = [13.0466022491, -10.525103569]
        g4_off = [12.5871200562, -16.5315208435]


        if exit[1] == 1:

            rospy.loginfo("subgoal A1")

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = g1_off[0]
            goal.target_pose.pose.position.y = g1_off[1]
            goal.target_pose.pose.orientation.w = 1.0
            client.send_goal(goal)

            wait = client.wait_for_result()


        elif exit[1] == 2:

            rospy.loginfo("subgoal A2")

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = g2_off[0]
            goal.target_pose.pose.position.y = g2_off[1]
            goal.target_pose.pose.orientation.w = 1.0
            client.send_goal(goal)
            
            wait = client.wait_for_result()

        else:
            rospy.loginfo("index=0")






        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
