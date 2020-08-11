from scipy.io import loadmat
import numpy as np

def world2object(poses:np.ndarray)->np.ndarray: 
    #poses 0 dim means 15*3(joints location) and 1dim means frame number
    oCoor= np.zeros(poses.shape); #coordinate in object coordinate system for each frame
    lshld = 2; rshld = 5; wist = 8;
    njoints=poses.shape[0]/3;
    nframe=poses.shape[1]
    for i in range(nframe):

        wCoor=np.reshape(poses[:,i],(-1,3)).T #coordinate in world coordinate system for a 3d object

        xAxis=wCoor[:,rshld]-wCoor[:,lshld]
        norm2=np.linalg.norm(xAxis)
        xAxis=xAxis/norm2 #x axis's unit vector

        midShld=0.5*(wCoor[:,rshld]+wCoor[:,lshld])
        yAxis=midShld-wCoor[:,wist]
        norm2=np.linalg.norm(yAxis)
        yAxis=yAxis/norm2 

        zAxis=np.cross(xAxis,yAxis)
        norm2=np.linalg.norm(zAxis)
        zAxis=zAxis/norm2 

        tranMatrix=np.asarray([xAxis,yAxis,zAxis]).T
        tranMatrix=np.vstack((tranMatrix,[0,0,0]))
        tmprow=np.r_[midShld,1]

        tranMatrix=np.hstack((tranMatrix,np.reshape(tmprow,(4,1))))
        homoWCoor=np.vstack((wCoor,np.ones((1,int(njoints)),dtype=np.float64)))
        t=np.dot(np.linalg.inv(tranMatrix),homoWCoor)
        t=np.reshape(t[0:3,:].T,(-1));
        oCoor[:,i]=t

    return oCoor
    
# not a pure function
def makeMeanAsOrigin(poses:np.ndarray,dim=3)->np.ndarray:
    nframes=poses.shape[1]
    for i in range(nframes): # for every frame
        p=np.reshape(poses[:,i],(-1,dim)).T
        center=np.mean(p,axis=1)
        center=np.expand_dims(center,axis=1)
        p=p-np.repeat(center,p.shape[1],axis=1)
        p=np.reshape(p.T,(-1));
        poses[:,i]=p
    return poses
def scaleDown(pose):
        scale=np.sqrt(np.sum(pose*pose,axis=0))
        scale=np.expand_dims(scale,axis=0)
        pose=pose/np.repeat(scale,pose.shape[0],axis=0)
        return pose,scale

        

class datasetMat:
    def __init__(self,filename:str):
        self._matData:dict =loadmat(filename)

    def getPose3DNormalized(self):
        pose3Ds:np.ndarray = self._matData["pose3Ds"]; #45*812
        pose3Ds:np.ndarray=world2object(pose3Ds)
        pose3Ds:np.ndarray=makeMeanAsOrigin(pose3Ds,dim=3)
        pose3Ds,scale=scaleDown(pose3Ds)
        return pose3Ds,scale

    def getPose2DNormalized(self):
        pose2Ds:np.ndarray = self._matData["pose2Ds"]; #30*812
        pose2Ds:np.ndarray=makeMeanAsOrigin(pose2Ds,dim=2)
        pose2Ds,scale=scaleDown(pose2Ds)
        return pose2Ds,scale

    def getPose3Ds(self):
        pose3Ds:np.ndarray = self._matData["pose3Ds"]; #45*812
        return pose3Ds

    def getBase(self):
        B=loadmat("datasets/BaseMatrix.mat")
        B=B['B']
        init_poses=B['init_poses']
        return B,init_poses


CMUDataSet=datasetMat("datasets/S1_Box_1_C1.mat")
