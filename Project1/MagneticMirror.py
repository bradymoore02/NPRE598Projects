import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bfield
import ode
from scipy.integrate import odeint

#### Inputs ####
L= 0.5 # Distance between coils
# Limits on ion trajectories
Rmax = 1
Xmin = -L
Xmax = 2*L

# How many grid points to use
Xsteps = 11
Rsteps = 11

# Constants
q = 1.602e-19
m = 1.6726e-27
qm = q/m

# Create linspace in R and Z dimensions
Xvalues = np.linspace(Xmin,Xmax,Xsteps)
Rvalues = np.linspace(0,Rmax,Rsteps)

# Initial Velocities
vx0 = 2750
vy0 = 3000
vz0 = 0

# Current loop properties
Ra = 0.1 # Loop Radius [m]
I0 = 1000 # Loop current [A]
Nturns = 1 # number of turns in the loop

# set initial position
X0 = np.array([L/2,0,0,vx0,vy0,vz0])

# Time domain for integration
time = np.linspace(0,0.004,9999t()
                   8

def mirrorBgrid(X,R):
    center1 = np.array([0,0,0])
    center2 = np.array([L,0,0])
    euler = np.array([90,0,0])*np.pi/180.0
    BB = np.zeros([2,np.size(X),np.size(R)])
    for i, x in enumerate(X):
        for j, r in enumerate(R):
            point = np.array([x,r,0])
            Bx1,By1,Bz1 = bfield.loopxyz(Ra, I0, Nturns, center1, euler, point)
            Bx2,By2,Bz2 = bfield.loopxyz(Ra, I0, Nturns, center2, euler, point)
            BB[0][i][j] = Bx1+Bx2
            BB[1][i][j] = By1+By2
    return BB


def interpolator(x,y,z,xx,rr,BB):
    R = np.sqrt(np.power(y,2)+np.power(z,2))
    if y != 0:
        phi = np.arctan(z/y)
    elif z != 0:
        phi = np.pi/2*abs(z)/z
    else:
        phi = 0
    #print(phi)
    center1 = np.array([0,0,0])
    center2 = np.array([L,0,0])
    euler = np.array([90,0,0])*np.pi/180.0
    point = (x,y,z)
    x1,y1,z1 = bfield.loopxyz(Ra, I0, Nturns, center1, euler, point)
    x2,y2,z2 = bfield.loopxyz(Ra, I0, Nturns, center2, euler, point)
    print(x1,y1,z1)
    return x1+x2, np.cos(phi)*(y1+y2), np.sin(phi)*(y1+y2)

    X = x
    dR = rr[1][0]-rr[0][0]
    dX = xx[0][1]-xx[0][0]
    j = int(np.floor(R/dR))
    i = int(np.floor(X/dX))
    #print(f"x: {x}, y: {y}, z: {z}, i(R): {i}, j(Z): {j}")
    Rj = j*dR
    Xi = i*dX

    # Areas
    A0 = (Xi+dX-X) * (Rj+dR-R)
    A1 = (X-Xi) * (Rj+dR-R)
    A2 = (X-Xi)*(R-Rj)
    A3 = (Xi+dX-X) * (R-Rj)
    At = dR*dX
    # Weights
    w0 = A0/At
    w1 = A1/At
    w2 = A2/At
    w3 = A3/At
    try:
        xret = w0*BB[0][i][j]+w1*BB[0][i+1][j]+w2*BB[0][i+1][j+1]+w3*BB[0][i][j+1]
        rret = w0*BB[1][i][j]+w1*BB[1][i+1][j]+w2*BB[1][i+1][j+1]+w3*BB[1][i][j+1]
    except:
        print("out of bounds")
        return 0,0,0
    #return xret, yret, zret
    return xret, np.cos(phi)*rret, np.sin(phi)*rret

xx, rr = np.meshgrid(Xvalues, Rvalues)
BB = mirrorBgrid(Xvalues,Rvalues)
print(interpolator(0,.5,.2,xx,rr,BB))

def newtonLorentz(t, X):
    x, y, z, vx, vy, vz = X
    # E-field [V/m]
    Ex, Ey, Ez = (0,0,0)
    # B-field [T]
    #Bx, By, Bz = (0,0,0)
    Bx, By, Bz = interpolator(x,y,z,xx,rr,BB)

    print(Bx,By,Bz,x,y,z)
    # Newton-Lorentz equation in Cartesian coordinates
    Xdot = np.zeros(6)
    Xdot[0] = vx
    Xdot[1] = vy
    Xdot[2] = vz
    Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
    Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
    Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
    return Xdot

Bmag = BB[0]
for r in range(len(Rvalues)):
    for x in range(len(Xvalues)):
        Bmag[x][r] =  np.sqrt(BB[0][x][r]**2+BB[1][x][r]**2)
plt.contourf(np.transpose(xx),np.transpose(rr),(Bmag),50)
plt.contourf(np.transpose(xx),-np.transpose(rr),(Bmag),50)
plt.colorbar()
X = ode.rk4(newtonLorentz, time, X0)
x = []
y = []
z = []

for i in X:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
plt.plot(x,z)
plt.xlabel("x")
plt.ylabel("z")
plt.xlim(0,0.5)
plt.ylim(-.1,.1)
plt.show()
