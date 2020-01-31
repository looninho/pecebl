# -*- coding: utf-8 -*-
"""
Some basic functions for designing pattern.
"""
import numpy as np

π = np.pi

def sites_1(size,radius,a):
    '''create sites for PhotonicCrystal.
    
    Args:
        * size (float): size of the square shape containing the Pcs
        * radius (float): radius of the hole component
        * a (float): pitch between two holes
        
    return an array of positions: [x,y]
    '''
    
    h=a*np.sin(π/3) #space between 2 holes in y line
    nx=int((size-2*radius)/a);Lx=a*nx+2*radius #number of holes in x direction
    ny=int((size-2*radius)/h);Ly=h*ny+2*radius #number of holes in y direction
    sites = np.empty(((nx+1)*(ny+1),3))
    for i in range(sites.shape[0]):
        stepy=int(i/(nx+1))
        x = ((i % (nx+1)) + (stepy % 2)*0.5)*a -Lx/2 + radius
        y = stepy*h - Ly/2 + radius
        sites[i] = np.array([x,y])
    return sites

def sites_2(a, col_start=-12,col_end=12,col_step=1, row_start=-14,row_end=13,row_step=2):
    '''yet another sites for PhotonicCrystal.
    
    Args:
        * a (float): pitch between two sites
        * col_start (int): start column number (center is 0)
        * col_end (int): end column number
        * col_step (int): column step
        * row_start (int): start row number (center is 0)
        * row_end (int): end row number
        * row_step (int): row step
    
    return an array of positions: [x,y]
    '''
    
    cols=np.arange(col_start,col_end,col_step)
    rows=np.arange(row_start,row_end,row_step)
    site_points=[]
    h=a*np.sin(π/3)
    for row in rows[::-1]:
        for col in cols:
            yd = row*h
            xd = col*a
            yu = yd + h
            xu = xd + a/2
            site_points.append([xu,yu])
            site_points.append([xd,yd])
    return np.array(site_points)

def raithDots(p=1000,start_dose=1,dose_step=0.1,nx=30,ny=70):
    '''Raith Demo dots dose test: pitch=1 (default) or 0.5 µm.

    Args:
        * p (float): pitch between two dots
        * start_dose (float): first dose
        * dose_step (float): step of dose
        * nx (int): columns, number of dots in x axis
        * ny (int): lines, number of dots in y axis

    return an array of positions and dose: [x,y, dose]

    '''
    pattern = np.zeros_like(np.ndarray(((nx+1)*(ny+1),3)))
    for i in range(pattern.shape[0]):
        x = -nx*p/2 + (i%(nx+1))*p
        stepy=int(i/(nx+1))
        y = -ny*p/2 + stepy*p
        dose = start_dose + stepy*dose_step
        pattern[i]=[x,y,dose]
    return pattern

def dot(x,y,dose=1):
    '''just dot and dose.
    '''
    return np.array([[x,y,dose]])

def line(x1,y1,x2,y2,ss,dose=1):
    '''line start at (x1,y2), end at (x2,y2), step (ss) and dose at each step.
    '''
    d=np.sqrt((x2-x1)**2+(y2-y1)**2)
    a=np.arctan((y2-y1)/(x2-x1))
    pattern = dot(x1,y1,dose)
    for s in np.arange(ss,d+ss,ss):
        pattern = np.append(pattern, dot(x1+s*np.cos(a),y1+s*np.sin(a),dose), axis=0)
    return pattern

def rectangle(x1,y1,x2,y2,ss,dose=1):
    '''rectangle = rows * lines.
    '''
    pattern=line(x1,y1,x2,y1,ss,dose)
    for h in np.arange(ss,y2-y1+ss,ss):
        pattern=np.append(pattern,line(x1,y1+h,x2,y1+h,ss,dose),axis=0)
    return pattern

def ring(x,y,r,ss,dose=1):
    '''ring center (x,y), radius r, step size (ss) and dose.
    '''
    pattern = dot(x+r,y,dose)
    try:
        n=int(2*π*r/ss)
        a=2*π/n
        for ai in np.arange(a,2*π,a):
            pattern=np.append(pattern,dot(x+r*np.cos(ai),y+r*np.sin(ai),dose),axis=0)
    except ZeroDivisionError:
        a=0
    return pattern

def circle(x,y,r,ss,dose=1):
    '''circle filled center at (x,y), radius, step size (ss) and dose.
    '''
    pattern=dot(x,y,dose)
    try:
        nr=int(r/ss)
        for i in range(1,nr):
            pattern=np.append(pattern,ring(x,y,i*ss,ss,dose),axis=0)
    except ZeroDivisionError:
        nr=0
    return pattern

def triangle(x,y,a,ss,dose=1):
    '''todo!
    '''
    return

def poly(points,ss,dose=1):
    '''todo!
    '''
    return

def rot(alpha):
    '''rotation angle alpha.
    use case:
        l=linPattern(0,0,36,0,4)
        rota=rot(π/3)
        l2=l[:,:-1]*rota
    '''
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def append(pattern1,pattern2):
    '''append pattern2 to the end of pattern1.

    return a new array
    '''
    pout=np.append(pattern1,pattern2,axis=0)
    return pout

def move(x,y,pattern):
    '''move pattern to (x,y).

    return a new array
    '''
    pout=np.zeros_like(pattern)
    for i in range(pattern.shape[0]):
        pout[i,:-1]=[x,y]-pattern[i,:-1]
        pout[i,-1]=pattern[i,-1]
    return pout

def replace(points, ref_pattern):
    '''Replace each element in points by ref_pattern.

    return a new array
    '''
    pout=move(points[0,0],points[0,1],ref_pattern)
    for point in points[1:]:
        pout=np.append(pout,move(point[0],point[1],ref_pattern),axis=0)
    return pout

def example1(a=170, r=48, ss=4):
    '''photonic crystal example.
    
    Args:
        * a (float): pitch (nm) between two sites
        * r (float): hole radius (nm)
        * ss (flaot): step size (nm)
    
    return an array of positions: [x,y]
    '''
    a=170;r=48
    site_points=sites_2(a)
    local_pattern=circle(0,0,r,ss)
    i=0
    while  not np.allclose(site_points[i], [0.0,0.0], atol=1e-3):
        i += 1
    site_points=np.delete(site_points,i,0)
    final_pattern=replace(site_points,local_pattern)
    return final_pattern

def metasurface1(ss,w=[639.7,119.6],d=[428.6,133.4],bl_corner=[-122249.3,-122500],tr_corner=[120794.5,122500], nT=10):
    '''yet another pattern!
    
    Args:
        * ss (float): step size (nm)
        * w [float, float]:  two constituant widths (nm)
        * d [float, float]:  two distances of related constituant widths
        * bl_corner [float, float]:  the bottom-left corner coordinates
        * tr_corber [float, float]: the top_right corner coordinates
        * nT (optional) is the number of periods. if nT=0 full size is returned.
    Return the pattern for metasurface.
    '''
    T=sum(w)+sum(d) #the period
    L=T-d[-1]
    if nT != 0:
        W=(nT-1)*T;H=np.ceil(W)
    else:
        W=tr_corner[0]-bl_corner[0]
        H=tr_corner[1]-bl_corner[1]
    
    c = -L/2
    local_pattern = rectangle(c,-H/2,c+w[0],H/2,ss)
    c += w[0]
    for wi, di in zip(w[1:],d[:-1]):
        c += di
        rec = rectangle(c,-H/2,c+wi,H/2,ss)
        c += wi
        local_pattern = append(local_pattern, rec )
    
    N=np.round(W/T);T_p=W/N
    site_points=line(-W/2,0,W/2,0,T_p)
    
    final_pattern=replace(site_points,local_pattern)
    return final_pattern

def example2(ss=20):
    '''metasurface  example.

    Arg:
        * ss (float): step size (nm)
    
    return an array of positions: [x,y]
    '''
    return metasurface1(ss)

if __name__ == "__main__":
    from ..utils.utils import timer, plt

    r=48.0;a=170;ss=4
    start = timer()
    site_points=sites_2(a)
    local_pattern=circle(0,0,r,ss)

    print("sites shape: {}".format(site_points.shape))

    i=0
    # search dot's index and remove dot at (0,0):
    while  not np.allclose(site_points[i], [0.0,0.0], atol=1e-3):
        i += 1

    print("remove site[{}]".format(i))

    site_points=np.delete(site_points,i,0)
    print("new sites shape: {}".format(site_points.shape))

    #pattern=dotPattern(0,0)
    final_pattern=replace(site_points,local_pattern)
    dt = timer() - start
    print("total {} points created in {} s".format(final_pattern.shape[0], dt))

    plt.plot(final_pattern[:,0], final_pattern[:,1], 'o', ms=1)
    plt.axis('equal');plt.show()
