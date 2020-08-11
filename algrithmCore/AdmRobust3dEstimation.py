import numpy as np
import math
import spams
import cvxpy as cvx
from sympy import *
from sympy.vector import CoordSys3D, gradient
import os







def soft(w, T):
    sigmaReverse=1/(2*T)
    return np.multiply(np.sign(w), np.fmax(abs(w) - sigmaReverse, 0))

def update_R(m1,m2,X,H,Y,tao):
    T = tao/2
    M = np.vstack((np.transpose(m1), np.transpose(m2)))
    w = X - H/tao - M @ Y
    R = soft(w, T)
    return R

def update_m1(m1, m2, R, X, H, Y, xi, tao):
    x,y,z=symbols('x y z', real=True)
    m10 = np.mat([x, y, z]).T
    F0 = (Matrix(np.vstack((np.transpose(m10), np.transpose(m2))) @ Y + R - X + H/tao).norm(ord='fro'))**2 + Matrix((m10.T @ m2 + xi/tao)**2).norm()
    m1=[]

    m10=solve([diff(F0,x),diff(F0,y),diff(F0,z)],[x,y,z])
    m1.append(float(m10[x]))
    m1.append(float(m10[y]))
    m1.append(float(m10[z]))
    return np.mat(m1).T

def update_m2(m1, m2, R, X, H, Y, xi, tao):
    x,y,z=symbols('x y z', real=True)
    m20 = np.mat([x, y, z]).T
    F0 = (Matrix(np.vstack((np.transpose(m1), np.transpose(m20))) @ Y + R - X + H/tao).norm(ord='fro'))**2 + Matrix((np.transpose(m1) @ m20 + xi/tao)**2).norm()
    m2=[]
    m20=solve([diff(F0,x),diff(F0,y),diff(F0,z)],[x,y,z])
    m2.append(float(m20[x]))
    m2.append(float(m20[y]))
    m2.append(float(m20[z]))
    return np.mat(m2).T

def update_r(M, B, x, a, u, v1,N):
    T=N/2
    w=x-M@B@a-v1/N
    return soft(w,T)


def update_b(O, v2, a, N):
    T=N/2/0.05
    w = v2/N+ a
    b = soft(w, T)
    return b

class update_a:
    def __init__(self, B, M, r, x, u, v1, b, v2, N,C,L):
        self.B = np.mat(B)
        self.M = np.mat(M)
        self.r = np.mat(r)
        self.x = np.mat(x)
        self.u = np.mat(u)
        self.v1 = np.mat(v1)
        self.b = np.mat(b)
        self.v2 = np.mat(v2)
        A = np.transpose(self.B) @ np.transpose(self.M) @ self.M @ self.B
        A = A + np.eye(A.shape[0], dtype=float)
        a = 2 * (np.transpose(self.r - self.x + self.M @ self.u + self.v1 / N) @ self.M @ self.B - self.b.T + self.v2.T / N)
        w1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        w2 = np.hstack((a, np.zeros((a.shape[0], 1))))
        w = np.vstack((w1, w2))
        self.w = np.mat(w)
        self.Omega=[]
        m=len(C)
        for i in range(m):
            self.Omega.append(np.vstack((np.hstack((np.transpose(self.B) @ np.transpose(C[i]) @ C[i] @ self.B, np.transpose(self.B) @ np.transpose(C[i]) @ C[i] @ self.u)), np.hstack((np.transpose(self.u) @ np.transpose(C[i]) @ C[i] @ self.B, np.transpose(self.u) @ np.transpose(C[i]) @ C[i] @ self.u - L[i])))))


    def L2(self, P, Q, G, deta):
        L = cvx.trace(self.w @ Q) + cvx.trace(np.transpose(G) @ (Q - P)) + (deta / 2) * cvx.norm(Q - P, p='fro', axis=None) ** 2
        return L

    def update_Q(self, P, G, deta, q, m):
        Q = cvx.Variable((q,q),symmetric=True)
        constraints = []
        for i in range(m):
            constraints += [cvx.trace(self.Omega[i] * Q) == 0]
        constraints += [Q[q-1,q-1]==1]
        L2 = self.L2(P, Q, G, deta)
        obj = cvx.Minimize(L2)
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.SCS, verbose=False, eps=1e-4, alpha=1.8, max_iters=100000, scale=5, use_indirect=True)
        #print(prob.status)
        return Q.value

    @staticmethod
    def update_P(Q, G, deta):
        Q1 = Q + (2 / deta) * G
        S = (Q1 + np.transpose(Q1)) / 2
        a, b = np.linalg.eig(S)
        b = b.T
        v = max(a)
        k = list(a).index(v)
        v1 = np.mat(b[k]).T
        P = max(v, 0) * v1 * np.transpose(v1)
        return P

    def get_Q(self, P, G, deta, q, m, C, L):
        k = 0
        Q=np.zeros((q,q))
        while True:
            Q1=Q
            P1=P
            Q = self.update_Q(P, G, deta, q, m)
            P = self.update_P(Q, G, deta) 
            G = G + 0.5* (Q - P)
            k = k + 1
            if k>10 or max(np.linalg.norm(P - Q, ord = 2),np.linalg.norm(Q-Q1,ord=2),np.linalg.norm(P-P1,ord=2))<0.01:
                return Q

    @staticmethod
    def get_a(Q,q):
    	#r = Q.shape[0]
    	#a = Q[r-1]
    	#a = np.mat(a[:-1]).T
    	#a = a*(1/Q[q-1,q-1])
    	#return a
        S = (Q + np.transpose(Q)) / 2
        D, V = np.linalg.eig(S)
        maxval=max(D)
        maxind=list(D).index(maxval)
        z=math.sqrt(maxval)*V[:,maxind]
        a=np.mat(z[:q-1]).T/z[q-1]
        return a




class AdmRobust3dEstimation:
    def __init__(self):
        pass

    def CameraEstimation(self,X,Y,m1,m2):
        H = np.random.rand(2, 15)
        xi = 0.3
        tao = 20
        k=0
        T=[]
        W=[]
        R=H
        k=0
        while True:
            m = m1
            m3 = m2
            R1=R
            R = update_R(m1, m2, X, H, Y,tao)
            m1 = update_m1(m1, m2, R, X, H, Y, xi, tao)
            m2 = update_m2(m1, m2, R, X, H, Y, xi, tao)
            H = H + tao * (np.vstack((np.transpose(m1), np.transpose(m2))) @ Y + R - X)
            xi = xi + tao * np.transpose(m1) @ m2
            W.append(np.linalg.norm(R, ord=1)-np.linalg.norm(R1, ord=1))
            k=k+1
            if k==10 or abs(W[-1]+np.linalg.norm(m1 - m, ord = 2) + np.linalg.norm(m2 - m3, ord = 2)) < 0.03:
                break
            else:
                continue
        return m1, m2

    def PoseEstimation(self,M,B,x,u,C,L):
        m = len(C)
        O = 0.01
        
        a = spams.lasso(np.asfortranarray(x-M@u),np.asfortranarray(M@B),return_reg_path = False,lambda1=0.001)
        a = a.toarray()
        idx = []
        for i in range(len(a)):
            if a[i,0] != 0:
                    idx.append(i)
        a = np.mat([a[i,0] for i in idx]).T
        B = np.mat([np.matrix.tolist(B.T)[i] for i in idx]).T

        q = len(a)+1

        G = 0.000001*np.random.rand(q,q)
        v1 =0*np.random.rand(30, 1)
        v2 =0*np.random.rand(q-1, 1)
        N = 0.5 
        deta = 20
        A=[[0,0,0]]
        k=0

        while True:
            a1 = a
            r = update_r(M, B, x, a, u, v1, N)
            b = update_b(O, v2, a, N)
            #z = np.hstack((b.T,np.mat([1]))).T
            z = np.hstack((a.T,np.mat([1]))).T
            P = z*z.T
            update = update_a(B, M, r, x, u, v1, b, v2, N, C, L)
            Q = update.get_Q(P, G, deta, q, m, C, L)
            a = update.get_a(Q,q)
            v1 = v1 + N/2* (r - x + M @ (B @ a + u))
            v2 = v2 + N/2* (a - b)
            A.append([a,r,b])
            k+=1
            #if k==20 or abs(A[-1]-A[-2])<0.001:
            if k==10 or (max(map(lambda x:np.linalg.norm(x[0]-x[1],ord=2),zip(A[-1],A[-2])))<0.01 and np.linalg.norm(abs(b-a),ord=2)<0.01):
                break
        return a,B

