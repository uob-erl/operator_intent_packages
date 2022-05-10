#!/usr/bin/env python
#from __future__ import division, print_function

'''
code for HIER-MI experiment (BASIC node)
HERE the modelBOIR class is being called
'''

import rospy
import numpy as np
import tf
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial import distance
from std_msgs.msg import Float32, Int8
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
import time
from BOIR_model_class import modelBOIR
from collections import deque


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

# callback function for navigational goal (clicked point)
def call_nav(msg):
    global x_nav, y_nav
    x_nav = msg.pose.position.x
    y_nav = msg.pose.position.y


# callback function for robot's coordinates (MAP FRAME)
def call_robot(message):
    global x_robot, y_robot
    x_robot = message.pose.pose.position.x
    y_robot = message.pose.pose.position.y


def getPrepared():
    # prepare transformation from g0(MAP FRAME) to g0 -> g0_new(ROBOT FRAME) Exit point
    G0_msg = PointStamped()
    G0_msg.header.frame_id = "map"
    G0_msg.header.stamp = rospy.Time(0)
    G0_msg.point.x = g0[0]
    G0_msg.point.y = g0[1]
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
    return G0_msg, G1_msg, G2_msg, G3_msg


def get_processed_goals(list0, list1, list2, list3):
    orientation_list = rotation  # convert from quaternion to RPY (the yaw of the robot in /map frame)
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    yaw_degrees = yaw * 180 / np.pi
    rot = yaw_degrees  # rotation angle of the robot = yaw in ROBOT FRAME
    g0_new = [list0.point.x, list0.point.y]  #
    g1_new = [list1.point.x, list1.point.y]  # NEW coordinates' goals after transformations (ROBOT FRAME)
    g2_new = [list2.point.x, list2.point.y]  #  -//-
    g3_new = [list3.point.x, list3.point.y]  #  -//-
    new_goals = [g0_new[0], g0_new[1], g1_new[0], g1_new[1], g2_new[0], g2_new[1], g3_new[0], g3_new[1]] # list of set of goals (ROBOT FRAME)
    new = np.array(new_goals) # array --> useful for angle computation
    return new


def get_transformation(G0_msg, G1_msg, G2_msg, G3_msg):
    while True:
        try:
            (translation, rotation) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list0 = listener0.transformPoint("/base_link", G0_msg)  # transform g0 to base_link (ROBOT FRAME) , returns x,y
            list1 = listener1.transformPoint("/base_link", G1_msg)  # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener2.transformPoint("/base_link", G2_msg)  # transform g2 to base_link (ROBOT FRAME) , returns x,y
            list3 = listener3.transformPoint("/base_link", G3_msg)  # transform g3 to base_link (ROBOT FRAME) , returns x,y
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        new = get_processed_goals(list0, list1, list2, list3)  # get the new goals (i.e. new coordinates) in ROBOT FRAME
        return new


def calc_path(targets, robot_coord):
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
        srv.tolerance = 0.3  #0.5
        resp = get_plan(srv.start, srv.goal, srv.tolerance)
        rospy.sleep(0.06) # old 0.1 x 4 = 0.40 sec , 0.06 x 4 = 0.24 sec
        length = np.append(length, path_length)
    path = length
    #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    #print('path: ', path)
    return path


def calc_angle(new): # assume that robot's base is always (0,0)
    xs = []
    ys = []
    it = iter(new)
    [(xs.append(i), ys.append(next(it))) for i in it]
    angles = np.arctan2(ys, xs) * 180 / np.pi
    Angle = abs(angles)
    print("Angles: ", Angle)
    return Angle


def checkInside(x_center, y_center, radius, Xpoint, Ypoint):
    if (((Xpoint - x_center) ** 2) + ((Ypoint - y_center) ** 2) <= radius**2):
        return True
    else:
        return False


# MAIN
if __name__=='__main__':
    try:
        # --------------------------- ROS necessary stuff --------------------------- #
        rospy.init_node('test_fcn')
        robot_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_robot)
        nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
        get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)
        sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call_len)
        intent_pub = rospy.Publisher('most_probable_goal', Int8, queue_size = 1)
        max_post_pub = rospy.Publisher('max_posterior', Float32, queue_size = 1)
        listener = tf.TransformListener()  # create tf.TransformListener objects
        listener0 = tf.TransformListener()
        listener1 = tf.TransformListener()
        listener2 = tf.TransformListener()
        listener3 = tf.TransformListener()

        # --------------------------- Initialize parameters --------------------------- #
        area_START = [2.558259, -0.0879769]
        area_A = [21.1191673, 1.42768478]
        area_B = [14.18952274, -9.89095687]
        area_FINAL = [4.798, -13.401]

        area_A_circle = [26.67539, 4.758863]
        area_B_circle = [16.818109, -16.0262184]

        x_robot = y_robot = x_nav = y_nav = path_length = post_sum = post_avg = 0
        cnt = 1
        translation = rotation = list0 = list1 = list2 = list3 = []  # useful for transormation
        threshold_click = 7.0
        radiusA = 8.0
        radiusB = 9.0
        wA = 0.6
        wP = 0.4
        maxA = 180           # max value the angle can take
        maxP = 40            # max value the path can take (depends on the arena)
        n_goals = 4
        Delta = 0.2          # factor that determines CPT below
        number = 5
        data_cpt = np.ones((n_goals, n_goals)) * (Delta / (n_goals-1))
        np.fill_diagonal(data_cpt, 1-Delta)
        stored_values = []
        alpha = 0.15
        #stored_values = deque(maxlen=number+1)

        done = rospy.is_shutdown()
        rate = rospy.Rate(2)  # 0.8 Hz = 1.25 seconds
        while not done:
            start_time = time.time()
            robot_coord = [x_robot, y_robot]  # from amcl pose
            operator_click = [x_nav, y_nav]   # from move base simple goal
            check_area_FINAL = distance.euclidean(operator_click, area_FINAL)

            if check_area_FINAL < threshold_click:  # here starts the experiment
                print("Operator clicked in FINAL AREA -- Progressing towards FINAL")
                circle_area_A = checkInside(area_A_circle[0], area_A_circle[1], radiusA, robot_coord[0], robot_coord[1])
                circle_area_B = checkInside(area_B_circle[0], area_B_circle[1], radiusB, robot_coord[0], robot_coord[1])

                if circle_area_A:
                    print("I have approached areaA --- local BOIR here")
                    g0 = [19.824, -1.078]
                    g1 = [19.042, 9.397]
                    g2 = [26.187, 9.155]
                    g3 = [25.094, -2.833]
                    prior = np.ones(n_goals) * 1/n_goals
                    local_targets_areaA = [g0, g1, g2, g3]
                    G0_msg, G1_msg, G2_msg, G3_msg = getPrepared()  # prepare goals for transformation
                    while circle_area_A:
                        start_time = time.time()
                        print("I have approached areaA --- local BOIR here")
                        robot_coord = [x_robot, y_robot]  # from amcl pose
                        circle_area_A = checkInside(area_A_circle[0], area_A_circle[1], radiusA, robot_coord[0], robot_coord[1])
                        new = get_transformation(G0_msg, G1_msg, G2_msg, G3_msg)
                        path = calc_path(local_targets_areaA, robot_coord)
                        angle = calc_angle(new)
                        bayes = modelBOIR(path, angle, wA, wP, maxA, maxP, data_cpt, prior)
                        posterior = bayes.calc_posterior()
                        print("POSTERIOR: ", posterior)
                        print("maximum posterior (ghost excluded): ", np.max(posterior[1:]))
                        index = np.argmax(posterior[1:])

                        # EMA (Exponential Moving Average) attempts to correct for lag time by giving more weight to recent value moves than oder ones
                        if cnt <= number:
                            post_sum += np.max(posterior[1:])
                            post_avg = round(post_sum / cnt, 4)
                            print(post_avg)

                        elif cnt > number:
                            print("value to be published", post_avg)
                            post_avg = round(alpha * np.max(posterior[1:]) + (1-alpha) * post_avg, 4)
                            max_post_pub.publish(post_avg)

                        # stored_values.append(np.max(posterior[1:]))
                        # if cnt % number == 0 and cnt != 0:
                        #     print('iteration ', cnt)
                        #     print("stored_values ", stored_values)
                        #     max_post_avg = round(np.sum(stored_values) / len(stored_values), 4)
                        #     print("value to be published", max_post_avg)
                        #     max_post_pub.publish(max_post_avg)

                        print("most probable goal is: ", index+1)
                        prior = posterior


                        cnt += 1
                        print("--- %s seconds ---" % (time.time() - start_time))
                        print('---------------------------------------------------------------------------')
                        rate.sleep()

                elif circle_area_B:
                    print("I have approached areaB --- local BOIR here")
                    g0 = [13.744, -6.229]
                    g1 = [20.293, -10.421]
                    g2 = [16.260, -14.101]
                    g3 = [8.235, -13.906]
                    prior = np.ones(n_goals) * 1/n_goals
                    local_targets_areaB = [g0, g1, g2, g3]
                    G0_msg, G1_msg, G2_msg, G3_msg = getPrepared()  # prepare goals for transformation
                    while circle_area_B:
                        start_time = time.time()
                        print("I have approached areaB --- local BOIR here")
                        robot_coord = [x_robot, y_robot]  # from amcl pose
                        circle_area_B = checkInside(area_B_circle[0], area_B_circle[1], radiusB, robot_coord[0], robot_coord[1])
                        new = get_transformation(G0_msg, G1_msg, G2_msg, G3_msg)
                        path = calc_path(local_targets_areaB, robot_coord)
                        angle = calc_angle(new)
                        bayes = modelBOIR(path, angle, wA, wP, maxA, maxP, data_cpt, prior)
                        posterior = bayes.calc_posterior()
                        print("POSTERIOR: ", posterior)
                        print("maximum posterior (ghost excluded): ", np.max(posterior[1:]))
                        index = np.argmax(posterior[1:])

                        # EMA (Exponential Moving Average) attempts to correct for lag time by giving more weight to recent value moves than oder ones
                        if cnt <= number:
                            post_sum += np.max(posterior[1:])
                            post_avg = round(post_sum / cnt, 4)
                            print(post_avg)

                        elif cnt > number:
                            print("value to be published", post_avg)
                            post_avg = round(alpha * np.max(posterior[1:]) + (1-alpha) * post_avg, 4)
                            max_post_pub.publish(post_avg)

                        # stored_values.append(np.max(posterior[1:]))
                        # if cnt % number == 0 and cnt != 0:
                        #     print('iteration ', cnt)
                        #     print("stored_values ", stored_values)
                        #     max_post_avg = round(np.sum(stored_values) / len(stored_values), 4)
                        #     print("value to be published", max_post_avg)
                        #     max_post_pub.publish(max_post_avg)

                        print("most probable goal is: ", index+1)
                        prior = posterior

                        cnt += 1
                        print("--- %s seconds ---" % (time.time() - start_time))
                        print('------------------------------------------------------------')
                        rate.sleep()

                else:
                    print("---GLOBAL BOIR should be running---")
                    g0 = area_START
                    g1 = area_A
                    g2 = area_B
                    g3 = area_FINAL
                    prior = np.ones(n_goals) * 1/n_goals
                    global_targets = [g0, g1, g2, g3]  # correspond to GLOBAL targets
                    G0_msg, G1_msg, G2_msg, G3_msg = getPrepared()  # prepare goals for transformation
                    while circle_area_A == False and circle_area_B == False:
                        start_time = time.time()
                        print("---GLOBAL BOIR should be running---")
                        robot_coord = [x_robot, y_robot]  # from amcl pose
                        circle_area_A = checkInside(area_A_circle[0], area_A_circle[1], radiusA, robot_coord[0], robot_coord[1])
                        circle_area_B = checkInside(area_B_circle[0], area_B_circle[1], radiusB, robot_coord[0], robot_coord[1])
                        new = get_transformation(G0_msg, G1_msg, G2_msg, G3_msg)
                        path = calc_path(global_targets, robot_coord)
                        angle = calc_angle(new)
                        bayes = modelBOIR(path, angle, wA, wP, maxA, maxP, data_cpt, prior)
                        posterior = bayes.calc_posterior()
                        print("POSTERIOR: ", posterior)
                        print("maximum posterior (ghost excluded): ", np.max(posterior[1:]))
                        index = np.argmax(posterior[1:])

                        # EMA (Exponential Moving Average) attempts to correct for lag time by giving more weight to recent value moves than oder ones
                        if cnt <= number:
                            post_sum += np.max(posterior[1:])
                            post_avg = round(post_sum / cnt, 4)
                            print(post_avg)

                        elif cnt > number:
                            print("value to be published", post_avg)
                            post_avg = round(alpha * np.max(posterior[1:]) + (1-alpha) * post_avg, 4)
                            max_post_pub.publish(post_avg)

                        # stored_values.append(np.max(posterior[1:]))
                        # if cnt % number == 0 and cnt != 0:
                        #     print('iteration ', cnt)
                        #     print("stored_values ", stored_values)
                        #     max_post_avg = round(np.sum(stored_values) / len(stored_values), 4)
                        #     print("value to be published", max_post_avg)
                        #     max_post_pub.publish(max_post_avg)

                        print("most probable goal is: ", index+1)
                        prior = posterior

                        cnt += 1
                        print("--- %s seconds ---" % (time.time() - start_time))
                        print('------------------------------------------------------------')
                        rate.sleep()

            else:
                print("Unacceptable click --- TRY AGAIN")

            print("--- %s seconds ---" % (time.time() - start_time))
            print('------------------------------------------------------------')
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
