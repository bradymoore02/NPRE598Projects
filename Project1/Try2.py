import bfield
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import ode
from scipy.stats import kde


# Plot B-field first
def plotBfield(X,Y,Z,center1, center2, Ra, current):
    Bnorm = np.zeros((X.size,Y.size))
    point = np.zeros([3])
    for i in range(0,X.size):
      for j in range(0,Y.size):
        point[0] = X[i]
        point[1] = Y[j]
        point[2] = Z
        Bx, By, Bz = bfield.loopxyz(Ra, current, 1, center1, (0,0,0), point)
        Bx1, By1, Bz1 = bfield.loopxyz(Ra, current, 1, center2, (0,0,0), point)
        Bnorm[i][j] = np.sqrt((Bx+Bx1)**2 + (By+By1)**2 + (Bz+Bz1)**2)

    plt.figure(1)
    XX,YY = np.meshgrid(X,Y)
    plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
    plt.contourf(-np.transpose(XX),np.transpose(YY),Bnorm,30)

    plt.colorbar()

# Now solve Field Lines
def blines(y,x):
    X=y[0]
    Y=y[1]
    Z=y[2]
    direction=y[3]
    Bx, By, Bz = bfield.loopxyz(Ra, current, 1, center1, (0,0,0), np.array([X,Y,Z]))
    Bx1, By1, Bz1 = bfield.loopxyz(Ra, current, 1, center2, (0,0,0), np.array([X,Y,Z]))
    B = np.zeros([3])
    B[0] = Bx+Bx1
    B[1] = By+By1
    B[2] = Bz+Bz1
    Bnorm = np.sqrt((Bx+Bx1)**2 + (By+By1)**2 + (Bz+Bz1)**2)
    dY    = np.zeros(4)
    dY[0] = direction * B[0]/Bnorm
    dY[1] = direction * B[1]/Bnorm
    dY[2] = direction * B[2]/Bnorm
    dY[3] = 0.0
    return dY
# Initial position of the field line
def plotFieldLines():
    Nlines = 10
    fieldlines_X0     = np.linspace( 0, 0.15, Nlines )
    fieldlines_Y0     = np.linspace( 0.2,   0.2,  Nlines )
    fieldlines_Z0     =  np.linspace( 0,   0,  Nlines )
    fieldlines_direction = np.ones( Nlines )
    fieldlines_length = np.ones( Nlines ) * 0.3


    for i in range(np.size(fieldlines_X0,0)):
        # Top portion
        Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],fieldlines_direction[i]])
        interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
        fieldlines = odeint(blines, Y0, interval )
        #print (fieldlines)
        plt.plot( fieldlines[:,0], fieldlines[:,1], 'r-' )
        plt.plot( -fieldlines[:,0], fieldlines[:,1], 'r-' )

        # Bottom portion
        Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],-fieldlines_direction[i]])
        interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
        fieldlines = odeint(blines, Y0, interval )
        #print (fieldlines)
        plt.plot( fieldlines[:,0], fieldlines[:,1], 'r-' )
        plt.plot( -fieldlines[:,0], fieldlines[:,1], 'r-' )

Ra = 0.1
current = 1000
center1 = (0,0,0)
center2 = (0,.4,0)
# Constants
q = 1.602e-19
m = 1.6726e-27
qm = q/m
def sample(Xvalues, Yvalues):
    BB = np.zeros([2,np.size(Xvalues),np.size(Yvalues)])
    for i, x in enumerate(Xvalues):
        for j, y in enumerate(Yvalues):
            point = np.array([x,y,0])
            Bx1,By1,Bz1 = bfield.loopxyz(Ra, current, 1, center1, (0,0,0), point)
            Bx2,By2,Bz2 = bfield.loopxyz(Ra, current, 1, center2, (0,0,0), point)
            BB[0][i][j] = Bx1+Bx2
            BB[1][i][j] = By1+By2
    return BB

def interpolator(x,y,z):
    Bx, By, Bz = bfield.loopxyz(Ra, current, 1, center1, (0,0,0), np.array([x,y,z]))
    Bx1, By1, Bz1 = bfield.loopxyz(Ra, current, 1, center2, (0,0,0), np.array([x,y,z]))
    B = np.zeros([3])
    return Bx+Bx1, By+By1, Bz+Bz1

def newtonLorentz(t, X):
    x, y, z, vx, vy, vz = X
    # E-field [V/m]
    Ex, Ey, Ez = (0,0,0)
    # B-field [T]
    #Bx, By, Bz = (0,0,0)
    Bx, By, Bz = interpolator2(x,y,z)

    # Newton-Lorentz equation in Cartesian coordinates
    Xdot = np.zeros(6)
    Xdot[0] = vx
    Xdot[1] = vy
    Xdot[2] = vz
    Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
    Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
    Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
    return Xdot

def MaxwellBoltzmann(v, T):
    kB = 1.3807e-23
    m = 1.6726e-27
    return 4/np.sqrt(np.pi)*(m/2/kB/T)**(3/2)*v**2*np.exp(-m*v*v/(2*kB*T))

def bolt2(v, T):
    kB = 1.3807e-23
    m = 1.6726e-27
    return 2*(m/2/kB/T/np.pi)**(1/2)*np.exp(-m*v*v/(2*kB*T))

def randomV(cum,X, T):
    y =np.random.rand()
    index = np.argmin(abs(y-cum))
    v = X[index]
    return v
def randomAngle():
    return np.random.rand()*np.pi/2

# set constants for the integration part of the problem
current  = 1000
Ra = 0.1
X = np.linspace(  0.0, 0.2, 50 )
Y = np.linspace( -0.1, 0.5, 50 )
Z = 0.0
center1 = (0,0,0)
center2 = (0,.4,0)
plotBfield(X,Y,Z,center1,center2, Ra, current)
plotFieldLines()
# Constants
q = 1.602e-19
m = 1.6726e-27
qm = q/m

Xvalues = np.linspace(0,.2,500)
Yvalues = np.linspace(0,.4,500)
BB = sample(Xvalues,Yvalues)
def interpolator2(x,y,z):
    R = np.sqrt(x**2+z**2)
    Y = y
    if Y>.4 or Y<0:
        return 0,0,0
    if R>.1:
        return 0,0,0
    if x != 0:

        phi = np.arctan(z/x)
        if x <0:
            phi += np.pi
    elif z != 0:
        phi = np.pi/2*abs(z)/z
    else:
        phi = 0

    dY = Yvalues[1]-Yvalues[0]
    dR = Xvalues[1]-Xvalues[0]
    i = int(np.floor(R/dR))
    j = int(np.floor(Y/dY))
    #print(f"x: {x}, y: {y}, z: {z}, i(R): {i}, j(Z): {j}")
    Ri = i*dR
    Yj = j*dY

    # Areas
    A0 = (Ri+dR-R) * (Yj+dY-Y)
    A1 = (R-Ri) * (Yj+dY-Y)
    A2 = (R-Ri) * (Y-Yj)
    A3 = (Ri+dR-R) * (Y-Yj)
    At = dR*dY
    # Weights
    w0 = A0/At
    w1 = A1/At
    w2 = A2/At
    w3 = A3/At
    rret = w0*BB[0][i][j]+w1*BB[0][i+1][j]+w2*BB[0][i+1][j+1]+w3*BB[0][i][j+1]
    Yret = w0*BB[1][i][j]+w1*BB[1][i+1][j]+w2*BB[1][i+1][j+1]+w3*BB[1][i][j+1]
    return np.cos(phi)*rret, Yret, np.sin(phi)*rret
time = np.linspace(0,.0004, 500)
def trajectory(vx,vy,vz):
    vx0 = vx
    vy0 = vy
    vz0 = vz
    X0 = np.array([0,0.2,0,vx0,vy0,vz0])
    X = ode.rk4(newtonLorentz,time,X0)
    x = []
    y = []
    z = []
    for i in X:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    return x, y, z
# Set cumulative distribution function
T=293
X = np.linspace(-1e4,1e4,10000)

s=0
cum = np.zeros([len(X)])
print("grid done")
for i in range(len(X)):
    s += bolt2(X[i],T)
    cum[i] = s
xlost = []
ylost = []
xret = []
yret = []
for i in range(1000):
    print(f"particle {i}")
    vx = abs(randomV(cum,X, 293))
    vy = randomV(cum, X, 293)
    #angle = randomAngle()
    #vx = v*np.cos(angle)
    #vy = v*np.sin(angle)
    x,y,z = trajectory(vx,vy,0)
    lost = False
    if i%10==0:
        plt.plot(x,y)
    if max(y)>0.4 or min(y) < 0:
        lost = True
        xlost.append(vx)
        ylost.append(vy)
    else:
        xret.append(vx)
        yret.append(vy)
plt.show()
Bcenter = interpolator(0,0.2,0)
Bcoil = interpolator(0,0,0)
sintheta = np.sqrt(Bcenter[1]/Bcoil[1])
theta = np.arcsin(sintheta)
xxxx = [0,sintheta*8000]
yyyy = [0,np.cos(theta)*8000]
plt.plot(xxxx,yyyy, c='r', label="Theoretical Loss Cone")
plt.plot(np.array(xxxx),-np.array(yyyy), c='r')
plt.plot(xret,yret,marker='.', ls='none', c='k', label="Trapped Ions")
plt.legend()
plt.plot(np.array(xlost),np.array(ylost),marker='x', ls='none', c='b', label="Escaped Ions")
plt.xlabel('$V_\perp$')
plt.ylabel('$V_{||}$')



nbins=30
k = kde.gaussian_kde([xret,yret])
xi, yi = np.mgrid[0:5000:nbins*1j, -5000:5000:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.contourf(xi, yi, zi.reshape(xi.shape), cmap="Greys", levels = 4)
plt.colorbar()
plt.tight_layout()
# Change color palette
plt.show()
