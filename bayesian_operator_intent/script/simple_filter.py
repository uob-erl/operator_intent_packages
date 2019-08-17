#!/usr/bin/env python
#from __future__ import division, print_function

import rospy
import numpy as np
import random
import math
import tf
import actionlib
from actionlib import ActionServer
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial import distance
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from random import randint
# from filterpy.discrete_bayes import normalize
# from filterpy.discrete_bayes import update




x0 = 0.0
y0 = 0.0
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
gprime = Point()
g00 = Point()
g10 = Point()
g20 = Point()
g90 = Point()
g100 = Point()
g111 = Point()
Goal = PoseStamped()
Start = PoseStamped()




# callback function for navigational goal (g')
def call_nav(sms):
    global x_nav, y_nav, qx_nav, qy_nav, qz_nav, qw_nav
    x_nav = sms.pose.position.x
    y_nav = sms.pose.position.y
    qx_nav = sms.pose.orientation.x
    qy_nav = sms.pose.orientation.y
    qz_nav = sms.pose.orientation.z
    qw_nav = sms.pose.orientation.w


# callback function for robot's coordinates (MAP FRAME)
def call_rot(mes):
    global x0, y0, qx, qy, qz, qw
    x0 = mes.pose.pose.position.x
    y0 = mes.pose.pose.position.y
    qx = mes.pose.pose.orientation.x
    qy = mes.pose.pose.orientation.y
    qz = mes.pose.pose.orientation.z
    qw = mes.pose.pose.orientation.w


def call(path_msg):
    global path_length
    path_length = 0
    for i in range(len(path_msg.poses) - 1):
        position_a_x = path_msg.poses[i].pose.position.x
        position_b_x = path_msg.poses[i+1].pose.position.x
        position_a_y = path_msg.poses[i].pose.position.y
        position_b_y = path_msg.poses[i+1].pose.position.y

        path_length += np.sqrt(np.power((position_b_x - position_a_x), 2) + np.power((position_b_y - position_a_y), 2))




# compute likelihood : P(dist|goal) = normalized [e^(-k*dist)] , for all goals-distances
def compute_like(dist, Angle, m, k):
    out0 = np.exp(-k * dist) * np.exp(-m * Angle)
    like = out0 / np.sum(out0)
    return like


# compute conditional : normalized [SP(goal.t|goal.t-1) * b(goal.t-1)]
def compute_conditional(cond, prior):
    out1 =  np.matmul(cond, prior.T)
    sum = out1 / np.sum(out1)
    return sum


# compute posterior P(goal|theta) = normalized(likelihood * conditional)
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)
    return post




def run():

    rospy.init_node('bayesian_filter')

    get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)


    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()



    # create tf.TransformListener objects
    listener = tf.TransformListener()
    listenerNAV = tf.TransformListener()
    listener00 = tf.TransformListener()
    listener10 = tf.TransformListener()
    listener20 = tf.TransformListener()
    listener90 = tf.TransformListener()
    listener100 = tf.TransformListener()
    listener111 = tf.TransformListener()

    # client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    # client.wait_for_server()

    # subscribers
    rob_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, call_rot)
    nav_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_nav)
    sub = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, call)


    # publishers
    pub = rospy.Publisher('possible_goal', Float32, queue_size=1)
    public = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)

    # declare some variables
    m = 0.1
    k = 0.3
    n = 4   # number of goals
    l = 0.75
    Delta = 0.2

    # Initialize Prior-beliefs according to goals' number
    # nav goal should have higher prior belief at t=0
    data0 = np.ones(n-1) * (1-l)/(n-1)   # P(gnav)=0.75 , P(g0)= ... , P(g1)= ...
    prior = data0
    prior = np.insert(prior, 0, l, axis=0)


    # creation of Conditional Probability Table 'nxn' according to goals & Delta
    data1 = np.ones((n, n)) * (Delta / (n-1))
    np.fill_diagonal(data1, 1-Delta)
    cond = data1


    rate=rospy.Rate(10)

    while not rospy.is_shutdown():

        offset = -0.5
        g_prime = [x_nav, y_nav]  # CLICKED point - g'
        g0 = [-2.83128278901, -3.62014215797]
        g1 = [-0.215494300143, -5.70071558441]
        g2 = [3.08670737031, -5.93227324436]
        # g3 = [6.94527384787, -11.717640655]
        # g4 = [8.51079199583, 4.698199058]
        # g5 = [9.00735322774, -0.677194482712]
        # g6 = [11.8716085357, -0.506166845716]
        # g7 = [14.9899956087, 5.20262059311]
        # g8 = [20.101826819, -8.62395824827]
        # g9 = [22.9649931871, 10.4371948525]
        # g10 = [25.5818735158, 15.74648043]
        # g11 = [27.8554619121, 6.78231736479]


        # prepare transformation from g_prime(MAP FRAME) to gprime -> g_prime_new(ROBOT FRAME)
        gprime_msg = PointStamped()
        gprime_msg.header.frame_id = "map"
        gprime_msg.header.stamp = rospy.Time(0)
        gprime_msg.point.x = g_prime[0]
        gprime_msg.point.y = g_prime[1]

        # prepare transformation from g0(MAP FRAME) to g00 -> g0_new(ROBOT FRAME)
        g00_msg = PointStamped()
        g00_msg.header.frame_id = "map"
        g00_msg.header.stamp = rospy.Time(0)
        g00_msg.point.x = g0[0]
        g00_msg.point.y = g0[1]

        # prepare transformation from g1(MAP FRAME) to g10 -> g1_new(ROBOT FRAME)
        g10_msg = PointStamped()
        g10_msg.header.frame_id = "map"
        g10_msg.header.stamp = rospy.Time(0)
        g10_msg.point.x = g1[0]
        g10_msg.point.y = g1[1]

        # prepare transformation from g2(MAP FRAME) to g20 -> g2_new(ROBOT FRAME)
        g20_msg = PointStamped()
        g20_msg.header.frame_id = "map"
        g20_msg.header.stamp = rospy.Time(0)
        g20_msg.point.x = g2[0]
        g20_msg.point.y = g2[1]

        # # prepare transformation from g2(MAP FRAME) to g90 -> g9_new(ROBOT FRAME)
        # g90_msg = PointStamped()
        # g90_msg.header.frame_id = "map"
        # g90_msg.header.stamp = rospy.Time(0)
        # g90_msg.point.x = g9[0]
        # g90_msg.point.y = g9[1]
        #
        # # prepare transformation from g2(MAP FRAME) to g100 -> g10_new(ROBOT FRAME)
        # g100_msg = PointStamped()
        # g100_msg.header.frame_id = "map"
        # g100_msg.header.stamp = rospy.Time(0)
        # g100_msg.point.x = g10[0]
        # g100_msg.point.y = g10[1]
        #
        # # prepare transformation from g2(MAP FRAME) to g111 -> g11_new(ROBOT FRAME)
        # g111_msg = PointStamped()
        # g111_msg.header.frame_id = "map"
        # g111_msg.header.stamp = rospy.Time(0)
        # g111_msg.point.x = g11[0]
        # g111_msg.point.y = g11[1]


        try:
            (trans, rot) = listener.lookupTransform('/base_link', '/map', rospy.Time(0)) # transform robot to base_link (ROBOT FRAME) , returns x,y & rotation
            list_nav = listenerNAV.transformPoint("/base_link", gprime_msg)
            list = listener00.transformPoint("/base_link", g00_msg)  # transform g0 to base_link (ROBOT FRAME) , returns x,y
            list1 = listener10.transformPoint("/base_link", g10_msg) # transform g1 to base_link (ROBOT FRAME) , returns x,y
            list2 = listener20.transformPoint("/base_link", g20_msg) # transform g2 to base_link (ROBOT FRAME) , returns x,y
            # list9 = listener90.transformPoint("/base_link", g90_msg) # transform g9 to base_link (ROBOT FRAME) , returns x,y
            # list10 = listener100.transformPoint("/base_link", g100_msg) # transform g10 to base_link (ROBOT FRAME) , returns x,y
            # list11 = listener111.transformPoint("/base_link", g111_msg) # transform g11 to base_link (ROBOT FRAME) , returns x,y
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):

            continue


        # convert from quaternion to RPY (the yaw of the robot in /map frame)
        orientation_list = rot

        # rotation angle of the robot = yaw in ROBOT FRAME
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        yaw_degrees = yaw * 180 / np.pi
        rospy.loginfo("yaw: %s", yaw_degrees)


        # declare NEW coordinates' goals after the transformations (ROBOT FRAME)
        g_prime_new = [list_nav.point.x, list_nav.point.y]
        g0_new = [list.point.x, list.point.y]
        g1_new = [list1.point.x, list1.point.y]
        g2_new = [list2.point.x, list2.point.y]
        # g9_new = [list9.point.x, list9.point.y]
        # g10_new = [list10.point.x, list10.point.y]
        # g11_new = [list11.point.x, list11.point.y]


        # NEW robot's coordinates after transformation (we don't care !) --- robot's x,y always 0 in ROBOT FRAME
        # robot = [trans[0], trans[1]]
        # rospy.loginfo("Robot_FRAME: %s", robot)

        # robot coordinates (MAP FRAME)
        robot_coord = [x0, y0]

        # goals coordinates (MAP FRAME)
        targets_FIRST = [g_prime, g0, g1, g2]
        #targets_SECOND = [g_prime, g9, g10, g11]
        #t_FIRST = [g_prime[0], g_prime[1], g0[0], g0[1], g1[0], g1[1], g2[0], g2[1]]
        #t_SECOND = [g9[0], g9[1], g10[0], g10[1], g11[0], g11[1]]

        # goals coordinates (ROBOT FRAME)
        new_goals_FIRST = [g_prime_new[0], g_prime_new[1], g0_new[0], g0_new[1], g1_new[0], g1_new[1], g2_new[0], g2_new[1]]
        news_FIRST = np.array(new_goals_FIRST)


        # computation of 'n' Euclidean distances (1st Observation) and store in array (MAP FRAME) -- SAME in all FRAMES
        measure = np.array([])
        for p in targets_FIRST:
            dist = distance.euclidean(robot_coord, p)
            measure = np.append(measure, dist)
        dist = measure

        # angles' computation between robot (x=0, y=0) & each goal (2nd Observation)
        robot_base = [0, 0]
        ind_pos_x = [0, 2, 4, 6]
        ind_pos_y = [1, 3, 5, 7]
        dx = news_FIRST - robot_base[0]
        dy = news_FIRST - robot_base[1]
        Dx = dx[ind_pos_x]
        Dy = dx[ind_pos_y]
        angle = np.arctan2(Dy, Dx) * 180 / np.pi
        angle = abs(angle)


        # BAYES' FILTER
        likelihood = compute_like(dist, angle, m, k)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        index = np.argmax(posterior)
        prior = posterior


        # define x,y of most likely goal according to index
        if index == 0:
            x_stoxos = g_prime[0] - offset
            y_stoxos = g_prime[1] - offset
        elif index == 1:
            x_stoxos = g0[0] - offset
            y_stoxos = g0[1] - offset
        elif index == 2:
            x_stoxos = g1[0] - offset
            y_stoxos = g1[1] - offset
        else:
            x_stoxos = g2[0] - offset
            y_stoxos = g2[1] - offset


        # # check euclidean distance(nav-g1) to define the proper set of goals (either 0,1,2 OR 9,10,11)
        # check = distance.euclidean(g_prime, g0)
        # rospy.loginfo("CHECK: %s", check)


        # generate a plan towards the most likely goal (--> path length)
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
        Goal.pose.position.x = x_stoxos
        Goal.pose.position.y = y_stoxos

        srv = GetPlan()
        srv.start = Start
        srv.goal = Goal
        srv.tolerance = 0.5
        resp = get_plan(srv.start, srv.goal, srv.tolerance)

        
        # print ...
        rospy.loginfo("NAV_goal: %s", g_prime)
        rospy.loginfo("path_length %s", path_length)
        rospy.loginfo("Distance: %s", dist)
        rospy.loginfo("Angles: %s", angle)
        rospy.loginfo("Posterior: %s", posterior)
        rospy.loginfo("Potential Goal is %s", index)









       #  stoxos = MoveBaseGoal()
       #  stoxos.target_pose.header.frame_id = "map"
       #  stoxos.target_pose.header.stamp = rospy.Time.now()
       # # Move 0.5 meters forward along the x axis of the "map" coordinate frame
       #  stoxos.target_pose.pose = Pose(Point(x_stoxos, y_stoxos, 0.000), Quaternion(0, 0, 0, 1))
       #
       # # Sends the goal to the action server.
       #  client.send_goal(stoxos)
       # # Waits for the server to finish performing the action.
       #  #wait = client.wait_for_result()
       # # If the result doesn't arrive, assume the Server is not available
       #  #client.get_result()
       #





        #public.publish(Goal)

        pub.publish(index)


        #rospy.Duration(5)

        rate.sleep()


if __name__=='__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
