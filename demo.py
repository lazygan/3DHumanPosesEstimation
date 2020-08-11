from scipy.io import loadmat,savemat
import numpy as np
import math
from scipy import linalg

import sys


def select_limb(limb:list,njoints:int):
    nlimb=len(limb)
    C=[np.zeros((3,int(3*njoints))) for i in range(nlimb)]
    for i in range(nlimb):
        E1=np.zeros((3,int(3*njoints)))
        E1[:,3*(limb[i,0]):3*(limb[i,0]+1)]=np.eye(3)
        E2=np.zeros((3,int(3*njoints)))
        E2[:,3*(limb[i,1]):3*(limb[i,1]+1)]=np.eye(3)
        C[i]=E1-E2
    return C

def lmb_length(y,C):
    nlimb=len(C)
    L=[None for i in range(nlimb)]
    for i in range(nlimb):
        L[i]=np.linalg.norm(np.dot(C[i],y))
    return list(map(lambda x: x**2,L))



model=loadmat("datasets/mytestmodel.mat")

B=model['B']
njoints=B.shape[0]/3
limb_ids = np.asarray([[3,4],[ 4,5],[ 6,7],[ 7,8],[ 10,11],[ 11,12],[13,14],[14,15]])-1
C=select_limb(limb_ids,njoints)

from datasets.datasetMat import  CMUDataSet
#pose3Ds, scale3=CMUDataSet.getPose3DNormalized(); 
#pose2Ds, scale2=CMUDataSet.getPose2DNormalized(); 
#model['pose2Ds']=pose2Ds;
#savemat("datasets/mytestmodel.mat",model)
#print("finished")

pose3Ds=model['pose3Ds']
pose2Ds=model['pose2Ds']
mu = np.zeros((pose3Ds.shape[0],1));

#y:3D姿态标记,x:对应的2D标记
index=int(sys.argv[1])
y=pose3Ds[:,index]
x=pose2Ds[:,index]

#实际实现中,需要指定肢体长度,此处假设已知肢体长度
L=lmb_length(y,C) 

X=np.reshape(x,(-1,2)).T
YReal=np.reshape(y,(-1,3)).T
x=np.expand_dims(x,axis=1) # change to column vector


from algrithmCore.AdmRobust3dEstimation import AdmRobust3dEstimation
admRobust3dEstimation=AdmRobust3dEstimation()

yout=model['init_pose']
youtarr=[[]]
youtarr.append(yout)

def CameraEstimation(X,Y):
      M=X@Y.T@np.linalg.inv(Y@Y.T)
      U,sigma,VT=np.linalg.svd(M)
      Raff=U@np.asarray([[1,0,0],[0,1,0]])@VT
      s=np.diag(sigma)
      if sum(abs(np.diag(U)))<=1 :
          s=np.rot90(s,2)
      M=s@Raff
      #M=X@Y.T@np.linalg.inv(Y@Y.T)
      #r,q=linalg.rq(M)
      #r[0,2]=0.05*r[0,2];
      #M=r@q
      return M
 
for i in range(10):
    Y=np.reshape(yout,(-1,3)).T
    print("Iteration",i)
    M=CameraEstimation(X,Y)
    #print("2D投影的均方差值:",np.linalg.norm((M@Y-X).flatten()))
    M=np.kron(np.eye(int(njoints)),M);
    a,B1=admRobust3dEstimation.PoseEstimation(M,B,x,mu,C,L)
    yout = B1 @ a + mu
    yout = np.asarray(yout).flatten()
    youtarr.append(yout)
    print("Y_diff:",np.linalg.norm(youtarr[-1]-youtarr[-2],ord=2))
    print("norm2 error:",np.linalg.norm(youtarr[-1]-y,ord=2))
    if np.linalg.norm(youtarr[-1]-youtarr[-2],ord=2)<0.05:
        break

#预测姿态与实际姿态均方差
yout=youtarr[-1]
print("估计3D姿态与实际姿态均方差")
print("norm2 error:",np.linalg.norm(y-yout,ord=2))


#3D姿态可视化
from util.viz import Viz
viz3d=Viz()
viz3d.appendToShow(np.asarray(y*100))
viz3d.appendToShow(np.asarray(yout*100))
viz3d.showAllPic()


