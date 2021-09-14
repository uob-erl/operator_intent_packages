#!/usr/bin/env python
#from __future__ import division, print_function

'''
Copyright (c) 2020-2021, Dimitris Panagopoulos
All rights reserved.
----------------------------------------------------
This script predicts/estimates/recognizes human operator's intent (i.e. most probable goal that operator are advancing
towards) using Bayesian calculus. The method presented here called BOIR (Bayesian Operator Intent Recognition)
and is part of the paper got published in IEEE SMC 2021, Melbourne, Australia, 17-20 October 2021

Panagopoulos, D. et al. (2021). "A Bayesian-Based Approach to Human Operator Intent Recognition in Remote Mobile Navigation". Manuscript has been accepted
for publication in 2021 IEEE International Conference on Systems, Man and Cybernetics (SMC)

It's the latest, most up-to-date file/script in the form of python Class providing the ability to be conveyed to future work !

Check the details on README.md to learn more about the research and the experiment's nature.
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

class Bayes(object):
    def __init__(self):
        rospy.init_node('BayesianEstimationClass_BOIR')
        self.rate = rospy.Rate(4)
        self.done = rospy.is_shutdown()
        self.listener = tf.TransformListener()  # create tf.TransformListener objects
        self.listener1 = tf.TransformListener()
        self.listener2 = tf.TransformListener()
        self.listener3 = tf.TransformListener()
        self.G1_msg = PointStamped()
        self.G2_msg = PointStamped()
        self.G3_msg = PointStamped()
        self.get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)
        self.sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.call_len)
        self.robot_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.call_robot)
        self.intent_pub = rospy.Publisher('most_probable_goal', Int8, queue_size = 1)
        self.path_length = 0
        self.x_robot = 0
        self.y_robot = 0
        self.g1 = [19.04143524, 9.39682674]
        self.g2 = [26.1866455, 9.15536022]
        self.g3 = [25.09380531, -2.8331837]
        self.targets = [self.g1, self.g2, self.g3]
        self.robot_base = [0, 0]
        self.robot_map = None
        self.wA = 0.6             # weighting factor for Angle
        self.wP = 0.4             # weighting factor for path
        self.maxA = 180           # max value the angle can take
        self.maxP = 25            # max value the path can take (depends on the arena)
        self.n_goals = 3          # number of total goals
        self.Delta = 0.2          # factor that determines CPT below
        self.prior = np.ones(self.n_goals) * 1/self.n_goals
        self.data_cpt = np.ones((self.n_goals, self.n_goals)) * (self.Delta / (self.n_goals-1))
        np.fill_diagonal(self.data_cpt, 1-self.Delta)
        self.transition = self.data_cpt

# --------------------------------- C A L L B A C K S --------------------------------- #
    def call_robot(self, message):
        #global x_robot, y_robot
        self.x_robot = message.pose.pose.position.x
        self.y_robot = message.pose.pose.position.y
        self.robot_map = [self.x_robot, self.y_robot]

    def call_len(self, path_msg):
        self.path_length = 0
        # loop through each adjacent pair of Poses in the path message
        for i in range(len(path_msg.poses) - 1):
            position_a_x = path_msg.poses[i].pose.position.x
            position_b_x = path_msg.poses[i+1].pose.position.x
            position_a_y = path_msg.poses[i].pose.position.y
            position_b_y = path_msg.poses[i+1].pose.position.y
            self.path_length += np.sqrt(np.power((position_b_x - position_a_x), 2) + np.power((position_b_y - position_a_y), 2))

# --------------------------------- P R O C E S S I N G   G O A L S --------------------------------- #
    def initPrep(self):
        # prepare transformation from g1(MAP FRAME) to g1 -> g1_new(ROBOT FRAME)
        self.G1_msg = PointStamped()
        self.G1_msg.header.frame_id = "map"
        self.G1_msg.header.stamp = rospy.Time(0)
        self.G1_msg.point.x = self.g1[0]
        self.G1_msg.point.y = self.g1[1]
        # prepare transformation from g2(MAP FRAME) to g2 -> g2_new(ROBOT FRAME)
        self.G2_msg = PointStamped()
        self.G2_msg.header.frame_id = "map"
        self.G2_msg.header.stamp = rospy.Time(0)
        self.G2_msg.point.x = self.g2[0]
        self.G2_msg.point.y = self.g2[1]
        # prepare transformation from g3(MAP FRAME) to g3 -> g3_new(ROBOT FRAME)
        self.G3_msg = PointStamped()
        self.G3_msg.header.frame_id = "map"
        self.G3_msg.header.stamp = rospy.Time(0)
        self.G3_msg.point.x = self.g3[0]
        self.G3_msg.point.y = self.g3[1]

    def processed_goals(self):
        orientation_list = self.rotation  # convert from quaternion to RPY (the yaw of the robot in /map frame)
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        yaw_degrees = yaw * 180 / np.pi
        rot = yaw_degrees  # rotation angle of the robot = yaw in ROBOT FRAME
        g1_new = [self.list1.point.x, self.list1.point.y]  # NEW coordinates' goals after transformations (ROBOT FRAME)
        g2_new = [self.list2.point.x, self.list2.point.y]  #  -//-
        g3_new = [self.list3.point.x, self.list3.point.y]  #  -//-
        new_goals = [g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]] # list of set of goals (ROBOT FRAME)
        self.new = np.array(new_goals) # array --> useful for angle computation
        #print(self.new)

    # def calc_distance(self):
    #     measure = np.array([])
    #     for x in self.targets:
    #         dis = distance.euclidean(self.robot_map, x)
    #         measure = np.append(measure, dis)
    #     dis = measure
    #     #print('euclidean', dis)

# --------------------------------- O B S E R V A T I O N   S O U R C E S --------------------------------- #
    def calc_angle(self):
        # angles computation between robot (x=0, y=0) & each transformed goal
        # if n=3 ..
        ind_pos_x = [0, 2, 4]
        ind_pos_y = [1, 3, 5]
        dx = self.new - self.robot_base[0]
        dy = self.new - self.robot_base[1]
        Dx = dx[ind_pos_x]
        Dy = dx[ind_pos_y]
        angles = np.arctan2(Dy, Dx) * 180 / np.pi
        self.Angle = abs(angles)
        #print('angles', self.Angle)

    def calc_path(self):
        self.length = np.array([])
        for j in self.targets:
            Start = PoseStamped()
            Start.header.seq = 0
            Start.header.frame_id = "map"
            Start.header.stamp = rospy.Time(0)
            Start.pose.position.x = self.robot_map[0]
            Start.pose.position.y = self.robot_map[1]
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
            resp = self.get_plan(srv.start, srv.goal, srv.tolerance)
            rospy.sleep(0.05) # 0.05 x 3 = 0.15 sec

            self.length = np.append(self.length, self.path_length)
        path = self.length
        print('path', path)

# --------------------------------- B A Y E S I A N   E S T I M A T I O N --------------------------------- #
    def calc_likelihood(self):
        a = self.Angle / self.maxA
        p = self.length / self.maxP
        self.c_l = np.exp(-a / self.wA) * np.exp(-p / self.wP)
        #print('likelihood', self.c_l)

    def calc_conditional(self):
        self.c_c = np.matmul(self.transition, self.prior.T)
        print('conditional', self.c_c)

    def calc_posterior(self):
        output = self.c_l * self.c_c
        self.c_p = output / np.sum(output)
        print('posterior', self.c_p)

    def maximum_and_recursion(self):
        index = np.argmax(self.c_p)
        self.prior = self.c_p
        print("most_probable_goal: %s", index+1)



    def run(self):
        while not self.done:
            start_time = time.time()
            self.initPrep()
            try:
                (self.translation, self.rotation) = self.listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
                self.list1 = self.listener1.transformPoint("/base_link", self.G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
                self.list2 = self.listener2.transformPoint("/base_link", self.G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
                self.list3 = self.listener3.transformPoint("/base_link", self.G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            self.processed_goals()
            #self.calc_distance()
            self.calc_angle()
            self.calc_path()
            self.calc_likelihood()
            self.calc_conditional()
            self.calc_posterior()
            self.maximum_and_recursion()
            print("--- %s seconds ---" % (time.time() - start_time))  # show me execution time for each iteration
            print('-----------------------------------------------------------------------')



# if __name__ == '__main__':
#   try:
#     bayes = Bayes()
#     bayes.run()
#   except rospy.ROSInterruptException:
#     rospy.loginfo("Test is complete")
