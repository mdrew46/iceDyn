"""
Finite volume ice sheet model with two accumulation zones
"""
import argparse
import time
import numpy as np
from copy import deepcopy
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as clr
import matplotlib.cm as cm

s=[]
def main():
    #######Constants######
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--num_cells", type=int, default=40, help="Number of grid cells")
    parser.add_argument("--num_steps", default=75000, help="Number of time steps taken over alotted simulation time")
    parser.add_argument("--diff_its", default=9, help="Number of iterations to solve for diffusion at each time step")
    parser.add_argument("--output", default="plot", help="output image file")
    args = parser.parse_args()
    nsteps=int(deepcopy(args.num_steps))
    numCells=int(deepcopy(args.num_cells))
    diffIts=int(deepcopy(args.diff_its))
    filename=str(deepcopy(args.output))
    #start=time.clock()
    f=1.00
    g=9.81
    rho=910.0
    Ao=1.0/31556926. * 1.0e-16
    a=0.3/31556926.
    L=1500.0e3/2.
    alpha = (5.*a)/(2.*(rho*g)**3. * Ao)
    Z=((5.*a*(L**4.))/( 2. * Ao * (rho*g)**3. ))**(1./8.)
    totTime=200000.0
    nyears=totTime/nsteps
    delT=nyears*31556926.0*a/Z
    delR=1.0/(numCells-1)
    divHeight = np.zeros(nsteps)
    ######heat stuff#####
    time=0
    colors=cm.Paired(np.linspace(0,1,nsteps))
    vol=np.zeros(nsteps)
    volmax=5.0e6
    ### Grid ###
    snowRadius=L
    d0=np.ones(numCells,dtype=np.float64)
    delR=1.0/(numCells)
    r=np.linspace(0., 1., numCells+1)[np.newaxis]
    r=r.T
    ##
    ### Source ###
    #omega=2*np.pi/period
    Sc=np.ones(numCells+1)
    fraction=0.20
    for i in range(0,numCells+1):
        if L*r[i]<=fraction*snowRadius : 
            Sc[i]=1.0
        elif L*r[i]>=(1-fraction)*snowRadius : 
            Sc[i]=1.0
        else:
            Sc[i]=-1.0
#    Sc=0.5+0.5*np.cos(omega*time)
#    Sc=1.0
    Sp=-0.00
    ### Time dependent stuff ###
    rho=1
    ap0=(rho*delR/delT)*np.ones(numCells+1,dtype=np.float64)
    global s
    s=np.zeros(numCells+1,dtype=np.float64)
    s0=np.zeros(numCells+1,dtype=np.float64)
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(r,s0,color=colors[0])
    ###Time iteration###
    b=np.zeros(numCells+1,dtype=np.float64)
    c=np.zeros(numCells+1,dtype=np.float64)
    d=np.zeros(numCells+1,dtype=np.float64)
    a=np.zeros(numCells+1,dtype=np.float64)
    for i in range(0,nsteps):
        ###Coefficients###
        d0=diffusion(r,s0,delR).reshape(numCells)
        for nits in range(0,diffIts):
            d0=diffusion(r,surface(r,s0,delR,Sc,Sp,d0,numCells+1,ap0,f),delR).reshape(numCells)
        s=surface(r,s0,delR,Sc,Sp,d0,numCells+1,ap0,f)
        s0=s
        time+=nyears
        for iterations in range(0,len(s)):
            vol[i]+=Z*delR*s[iterations]
        for k in range(0,numCells+1):
            if L*r[k]<=fraction*snowRadius : 
                Sc[k]=1.0-(vol[k]/volmax)
            elif L*r[k]>=(1-fraction)*snowRadius : 
                Sc[k]=1.0-(vol[k]/volmax)
            else:
                if s[k]>500.:
                    Sc[k]=0
                else:
                    Sc[k]=-1.0
        if i%1000 == 0:
            plt.plot(L*r,Z*s,color=colors[i])
#        Sc=0.5+0.5*np.cos(omega*time)
#        divHeight[i]=s[0]
#    plt.figure(1)
#    plt.plot(np.linspace(0,totTime,nsteps),Z*divHeight)

    ###################
    # Analytic solution
    ###################
    zs=2.**(3./8.) * (1/2.)**(1./8.) * ( 1**(4./3.) - (r)**(4./3.))**(3./8.)
#    plt.figure(1)
#    plt.plot(L*r,Z*zs,'r--',)
    ax=plt.gca()
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Surface Elevation (m)')
    plt.subplot(2,1,2)
    plt.plot(np.linspace(0.,1.,nsteps)*time,vol,'r--')
    plt.xlabel('time')
    plt.ylabel('total ice volume')
#    ax.set_ylim([0,2000])
    plt.savefig(args.output+"x"+str(numCells)+"t"+str(nsteps)+"dits"+str(diffIts)+".eps",format='eps', dpi=1024)
#    ani=animation.FuncAnimation(plt.figure(1),getS,nsteps)
#    ani.save('test.mp4',fps=30)

########################################################
################# Animate s function ###################
########################################################
#Not Working Yet
def getS(i):
    return [i]
########################################################
################ s calculation function ################
########################################################
def surface(r, s0, dr, Sc,Sp,d0,numCells, ap0, f):
    b=np.zeros(numCells,dtype=np.float64)
    c=np.zeros(numCells,dtype=np.float64)
    d=np.zeros(numCells,dtype=np.float64)
    a=np.zeros(numCells,dtype=np.float64)
    s=np.zeros(numCells,dtype=np.float64)
    c[1:]= d0[:]/dr#West
    b[:-1]=d0[:]/dr#East
    d= Sc*dr*np.ones(numCells,dtype=np.float64)
    d[0]= Sc[0]*dr
    d[-1]= Sc[-1]*dr

#    d= Sc*dr*np.ones(numCells,dtype=np.float64)+ap0*s0
    a= f*b+f*c-f*Sp*dr+ap0
    ###### Boundary Conditions #####
    #Left
    c[0]=0.0
    a[0]=f*b[0] + f*c[0] - f*Sp*dr + ap0[0] 
    #Right
    b[-1]=0.0
    a[-1]=f*b[-1]+f*c[-1]-f*Sp*dr+ap0[-1]
    for k in range(0,numCells):
        if a[k]<0 :
            print "a has negative at ",k
        elif b[k]<0 :
            print "b has negative at ",k
        elif c[k]<0 :
            print "c has negative at ",k
    MD=np.zeros((len(r),1))
    UD=np.zeros((len(r),1))
    LD=np.zeros((len(r),1))
    MD=a
    UD=-f*c
    LD=-f*b
    data=np.array([MD.reshape(len(r)),LD.reshape(len(r)),UD.reshape(len(r))])
    diags=np.array([0,-1,1])
    X=spdiags(data,diags,len(r),len(r)).toarray()
    MD=ap0-(1-f)*c-(1-f)*b-(1-f)*Sp*dr
    MD[0]=ap0[0]-(1-f)*c[0]-(1-f)*b[0]-(1-f)*Sp*dr
    MD[-1]=ap0[-1]-(1-f)*c[-1]-(1-f)*b[-1]+(1-f)*Sp*dr
    UD=(1-f)*c
    LD=(1-f)*b
    data=np.array([MD.reshape(len(r)),LD.reshape(len(r)),UD.reshape(len(r))])
    Y=spdiags(data,diags,len(r),len(r)).toarray()
    Ys=Y.dot(s0)
    right=Ys+d
    s=np.linalg.matrix_power(X,-1).dot(right)
    for i in range(0,len(s)):
        if s[i]<0:
            s[i]=0
#    s[-1]=0
    ###### Check Rule 2 (Pos Coeff) ######
    return s
########################################################
################# Diffusion function ##################
########################################################
def diffusion(radius, surface_elevation, delta_radius):
    diffus=np.zeros((len(radius)-1,1),dtype=np.float64)
    deriv=np.diff(surface_elevation)
    deriv=deriv/delta_radius
    aveSurf=(surface_elevation[1:]+surface_elevation[:-1])/2.
    aveRad=(radius[1:]+radius[:-1])/2.
    for i in  range (0,len(surface_elevation)-1):
        diffus[i]=deriv[i]*deriv[i]*aveSurf[i]*aveSurf[i]*aveSurf[i]*aveSurf[i]*aveSurf[i]*aveRad[i]
    return diffus
########################################################
################### RMS function #######################
########################################################
def windowed_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size,dtype=np.float64)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))
if __name__ == "__main__":
    main()
