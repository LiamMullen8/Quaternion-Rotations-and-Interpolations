import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def quat(theta,x,y,z):
    # unit quaternion in the x,y,z direction (axis of rotation)   
	norm = math.sqrt(x**2 + y**2 + z**2)
	Q = [math.cos(theta/2),
		 (x/norm)*math.sin(theta/2),
		 (y/norm)*math.sin(theta/2),
		 (z/norm)*math.sin(theta/2)]
	return Q

def quat_mult(p,q):
	t0 = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
	t1 = p[0]*q[1] + p[1]*q[0] - p[2]*q[3] + p[3]*q[2]
	t2 = p[0]*q[2] + p[1]*q[3] + p[2]*q[0] - p[3]*q[1]
	t3 = p[0]*q[3] - p[1]*q[2] + p[2]*q[1] + p[3]*q[0]
	return [t0,t1,t2,t3]

def quat_dot(p,q):
    # component-wise multiplication
    Q = p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3]
    return Q

def quat_rot(qi,p,q):
    # active rotation -> q*pq
    # passive rotation -> qpq*
	r = quat_mult(qi,p)
	Pp = quat_mult(r,q)
	return Pp

def quat_inv(q):
    # quaternion conjugate => negate imaginary components
	return [q[0],-1*q[1],-1*q[2],-1*q[3]]

	
def lerp(p,q,t):
    th = p[0]*(1-t) + q[0]*t
    x = p[1]*(1-t) + q[1]*t
    y = p[2]*(1-t) + q[2]*t
    z = p[3]*(1-t) + q[3]*t
    return [th,x,y,z]

def slerp(p,q,t):
    # p . q = cos(O) => O = arccos(p . q)
    O = math.acos(quat_dot(p,q))
    
    th = (p[0]*math.sin((1-t)*O) + q[0]*math.sin(t*O))/math.sin(O)
    x = (p[1]*math.sin((1-t)*O) + q[1]*math.sin(t*O))/math.sin(O)
    y = (p[2]*math.sin((1-t)*O) + q[2]*math.sin(t*O))/math.sin(O)
    z = (p[3]*math.sin((1-t)*O) + q[3]*math.sin(t*O))/math.sin(O)
    return [th,x,y,z]



fig=plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

## initial points
radius = 2
theta = np.linspace(0, 2*np.pi, 50)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
ax.scatter(x,y,theta)
points = [(0,i,j,t) for i,j,t in zip(x,y,theta)]

print("POINTS:")
for p in points:
    print(f"{p[1:]}")
###

#A = np.linspace(0, 12*np.pi/6, 100)
#ax.quiver(0,0,0,3,4,5)
#for a in A:
#    q = quat(a,3,4,5)
#    print(f"quaternion: {q[1:]}")
#    qi = quat_inv(q)
#    Xs=[]
#    Ys=[]
#    Zs=[]
#    for point in points:
#        g = quat_rot(qi,point,q)
#        Xs.append(g[1])
#        Ys.append(g[2])
#        Zs.append(g[3])
#
#    ax.plot(Xs,Ys,Zs)


 
Q1 = quat(np.pi/6,2,2,6)
Q2 = quat(11*np.pi/6,-2,-4,-2)

T = np.linspace(0,1,50)

for t in range(len(T)):
    Qn = slerp(Q1,Q2,T[t])
    print(f"Q{t}:{Qn}")   
    ax.quiver(0,0,0,Qn[1],Qn[2],Qn[3])
    
    Qni = quat_inv(Qn)
    Xs=[]
    Ys=[]
    Zs=[]
    for point in points:
        g = quat_rot(Qni,point,Qn)
        Xs.append(g[1])
        Ys.append(g[2])
        Zs.append(g[3])
        ax.plot(Xs,Ys,Zs)
        
plt.show()