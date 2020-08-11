import numpy as np
from sympy import *
from sympy.vector import CoordSys3D, gradient
import os
#import amc_parser
import scipy.io
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import spams
#import cvxpy as cvx



from datasets.datasetMat import  CMUDataSet

# The coordinate system of poses3Ds: 
# origin: (x_0,y_0,z_0) which present the mean value of  all joint's coordinate (x,y,z)
# x Axis: vector point from left shoulder to right shoulder
# y Axis: vector point from midpoint of left and right shoulder to waist
# z Axis: the cross product of x Axis and y Axis
poses3Ds, scale3=CMUDataSet.getPose3DNormalized(); 
poses2Ds, scale2=CMUDataSet.getPose2DNormalized(); 
#




param = { 'K' : 200, 'lambda1' : 0.01, 'iter' :300}

#not the same result compared to authors original source code 
B=spams.trainDL(np.asfortranarray(poses3Ds),**param)
init_pose = np.expand_dims(np.mean(poses3Ds, 1),axis=1);
print(init_pose.shape)

#
scipy.io.savemat('./datasets/BaseMatrix.mat', mdict={'init_pose':init_pose,'B':B})






