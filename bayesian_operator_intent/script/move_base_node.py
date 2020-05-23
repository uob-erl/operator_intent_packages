#!/usr/bin/env python
#from __future__ import division, print_function


import rospy
import numpy as np
import random
import math
import tf
import actionlib
import time
from actionlib import ActionServer
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

# PENDING = 0
# ACTIVE = 1
# DONE = 2
# WARN = 3
# ERROR = 4

loa = 0
index = 0
state = 0




def state_callback(mess):
    global state
    state = mess.data

# def loa_callback(msg):
#     global loa
#     loa = msg.data

def index_callback(message):
    global index
    index = message.data


def active_cb():
    rospy.loginfo("Action server is processing the goal")

def feedback_cb(feedback):
    rospy.loginfo("Current: "+str(index))

def done_cb(status, result):
    if status == 3:
        rospy.loginfo("Goal reached")
    if status == 2 or status == 8:
        rospy.loginfo("Goal cancelled")
    if status == 4:
        rospy.loginfo("Goal aborted")






def run():

    rospy.init_node('MoveBaseNode')

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    #pub = rospy.Publisher("move_base/goal", MoveBaseActionGoal, queue_size=1)

    #sub_loa = rospy.Subscriber("/loa", Int8, loa_callback)
    sub_index = rospy.Subscriber('/possible_goal', Float32, index_callback)
    sub_state = rospy.Subscriber('/current_area', Int8, state_callback)

    #cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)




    rate = rospy.Rate(4)



    while not rospy.is_shutdown():

        exit = [state, index]
        #rospy.loginfo("loa: %s", exit[0])
        rospy.loginfo("area: %s", exit[0])
        rospy.loginfo("probable goal: %s", exit[1])

        g1 = [20.9167137146, 12.6046304703]
        g2 = [33.3691520691, 11.5497436523]



        g3 = [12.8780117035, -10.7926425934]
        g4 = [14.0271482468, -17.25157547]



        if exit[0] == 1:

            if exit[1] == 1:
                rospy.loginfo("subgoal A1")

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = g1[0]
                goal.target_pose.pose.position.y = g1[1]
                goal.target_pose.pose.orientation.z = -0.7
                goal.target_pose.pose.orientation.w = 0.714

                client.send_goal(goal, done_cb, active_cb, feedback_cb)
                #wait = client.wait_for_result()
                time.sleep(5)
                # client.cancel_goal()
                #state_result = client.get_state()

            elif exit[1] == 2:
                rospy.loginfo("subgoal A2")

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = g2[0]
                goal.target_pose.pose.position.y = g2[1]
                goal.target_pose.pose.orientation.z = 0.7245
                goal.target_pose.pose.orientation.w = 0.69

                client.send_goal(goal, done_cb, active_cb, feedback_cb)
                #state_result = client.get_state()
                time.sleep(5)
                #client.cancel_goal()
                #client.get_result())

            else:
                rospy.loginfo("index=0")


        elif exit[0] == 2:

            if exit[1] == 1:
                rospy.loginfo("subgoal B1")

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = g3[0]
                goal.target_pose.pose.position.y = g3[1]
                goal.target_pose.pose.orientation.z = -0.01
                goal.target_pose.pose.orientation.w = 0.999

                client.send_goal(goal, done_cb, active_cb, feedback_cb)
                #wait = client.wait_for_result()
                time.sleep(5)
                # client.cancel_goal()
                #state_result = client.get_state()

            elif exit[1] == 2:

                rospy.loginfo("subgoal B2")

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = g4[0]
                goal.target_pose.pose.position.y = g4[1]
                goal.target_pose.pose.orientation.z = 0.999
                goal.target_pose.pose.orientation.w = -0.0236

                client.send_goal(goal, done_cb, active_cb, feedback_cb)
                #state_result = client.get_state()
                time.sleep(5)
                #client.cancel_goal()
                #client.get_result())

            else:
                rospy.loginfo("index=0")


        else:
            rospy.loginfo("no move base needed")






        rate.sleep()




if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
