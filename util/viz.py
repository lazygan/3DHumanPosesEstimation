import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


import numpy as np


class Viz:
    def __init__(self):
        self.poses=[]
    def appendToShow(self,pose):
        self.poses.append(pose)
    def showOnePic(self,pose,ax,lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
        njoints =int(pose.shape[0] / 3);
        parent = np.asarray([2,1,2,3,4,2,6,7,2,9,10,11,9,13,14])-1;
        LR=np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        pose=np.reshape(pose,(-1,3)).T #coordinate in world coordinate system for a 3d object
        for i in range(njoints):
            x, y, z = [np.array( [pose[j, i],pose[j,parent[i]]] ) for j in range(3)]
            ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
        RADIUS = 750 # space around the subject
        xroot, yroot, zroot =pose[0,11],pose[1,11],pose[1,11]
        #ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        #ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        #ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
        if add_labels:
          ax.set_xlabel("x")
          ax.set_ylabel("y")
          ax.set_zlabel("z")
        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])
        #ax.set_aspect('equal')
        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        # Keep z pane
        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)
        
    def showAllPic(self):
       # Visualize random samples
      import matplotlib.gridspec as gridspec
      # 1080p	= 1,920 x 1,080
      fig = plt.figure( figsize=(19.2, 10.8) )
      nshow=len(self.poses)
      nshow=math.sqrt(nshow)
      nrow=math.floor(nshow)
      ncol=math.ceil(nshow)
      gs1 = gridspec.GridSpec(nrow,ncol) # 5 rows, 9 columns
      gs1.update(wspace=-0.00, hspace=0.00) # set the spacing between axes.
      plt.axis('off')
      
      for index,p in enumerate(self.poses):
          ax1 = plt.subplot(gs1[index],projection='3d')
          self.showOnePic(p, ax1 )
      import time
      timeStr=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
      plt.savefig("output/"+timeStr+".png")
 

if __name__ == '__main__' :
    pass
