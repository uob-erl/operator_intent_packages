#!/usr/bin/env python
#from __future__ import division, print_function

'''
simplified version .. it's being called by the "hier_experiment_MAIN.py" script
'''

import rospy
import numpy as np


class modelBOIR:
    def __init__(self, Path, Angle, wA, wP, maxA, maxP, data_cpt, prior):
        self.Path = Path
        self.Angle = Angle
        self.wA = wA             # weighting factor for Angle
        self.wP = wP           # weighting factor for path
        self.maxA = maxA           # max value the angle can take
        self.maxP = maxP          # max value the path can take (depends on the arena)
        self.data_cpt = data_cpt
        self.prior = prior


    def calc_likelihood(self):  # computing likelihood
        a = self.Angle / self.maxA
        p = self.Path / self.maxP
        c_l = np.exp(-a / self.wA) * np.exp(-p / self.wP)
        return c_l


    def calc_conditional(self):
        c_c = np.matmul(self.data_cpt, np.transpose(self.prior))
        print("cond", c_c)
        return c_c


    def calc_posterior(self): # max posterior and publisher shoud be IGNORING ghost
        c_l = self.calc_likelihood()
        c_c = self.calc_conditional()
        output = c_l * c_c
        c_p = output / np.sum(output)
        return c_p
