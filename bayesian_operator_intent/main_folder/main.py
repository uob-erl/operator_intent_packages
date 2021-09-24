#!/usr/bin/env python
#from __future__ import division, print_function

'''
define the goals and let Bayesian estimation (BOIR) perform
only BOIR here .. !!NO BOIR-AIRM!!
'''

import rospy
from class_bayes import Bayes



if __name__ == '__main__':
    try:
        # set goals here for estimation (Scenario1 here)
        g1 = [24.4425258636, 12.7283153534] #g_left
        g2 = [29.08319664, 12.852309227] #g_center = HUMAN
        g3 = [33.7569503784, 12.5955343246] #g_right
        # # set goals here for estimation (Scenario2)
        # g1 = [4.60353183746, -13.2964735031] #gleft
        # g2 = [6.86671924591, -9.13561820984] #gcenter
        # g3 = [4.69921255112, -4.81423997879] #gright
        # # set goals here for estimation (Scenario3)
        # g1 = [21.9794311523, -11.4826393127] #gleft
        # g2 = [21.3006038666, -17.3720340729] #gcenter
        # g3 = [4.67607975006, -20.1855487823] #gright
        # # set goals here for estimation (Scenario4)
        # g1 = [13.384099, -0.070828]
        # g2 = [23.11985, -1.370935]
        # g3 = [29.208362, -0.728264]
        # g4 = [27.863958, -6.041914]
        # g5 = [20.085504, -7.524883]
        bayes = Bayes(g1, g2, g3) # add extra g4, g5 if scenario5 is executed
                                  # make modifications in class where necessary
        bayes.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test is complete")
