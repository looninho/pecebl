import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e #in Coulomb

from timeit import default_timer as timer

#playsound
duration = 1000  # millisecond
freq = 440  # Hz    
try:
    import winsound
except ImportError:
    import os
    def playsound(frequency,duration):
        #apt-get install beep
        os.system('beep -f %s -l %s' % (frequency,duration))
else:
    def playsound(frequency,duration):
        winsound.Beep(frequency,duration)

def get_pre_psf(sim_electrons, sim, speed_Mhz, beam_current, i_y=3, i_z=3):
    ''' return the scaled psf data with your hardware specifications.
        Args:
            * sim_electrons (int): the number of simulated electrons
            * sim (Casino instance): the instance of imported data
            * speed_Mhz (MHz): the electronic speed of the ebeam writer
            * beam current (pA): the beam current of the beamer (sem)
            * i_y (int): y-index of data where the beam hit the surface
            * i_z (int): z-index (or depth of resist) of data where you 
                want to work at.
    '''
    min_dose = beam_current*1e-18 / speed_Mhz
    nb_electrons = min_dose / e
    norm_fact = nb_electrons / sim_electrons
    r = np.arange(sim.xrange[0],sim.xrange[1],sim.sx)
    mcdata = np.c_[r, 1000*sim.data[:,i_y,i_z]*norm_fact]
    return mcdata

def idx2val(simdata,idx=0,axis=2):
    '''ouput the value belong to axis (0=x; 1=y; 2=z).
    usage: idx2val(sim.data,3,2).'''
    assert isinstance(simdata,np.ndarray)
    assert isinstance(idx, int)
    assert isinstance(axis, int)
    assert axis < 3 and axis >=0
    if len(simdata.shape) != 3:
        print("not a 3D-data!")
        return
    if idx >= simdata.shape[axis]:
        print("index out of range")
        return
    else:
        if axis == 0:
            return sim.xrange[0]+idx*sim.sx
        elif axis == 1:
            return sim.yrange[0]+idx*sim.sy
        else:
            return sim.zrange[0]+idx*sim.sz

def dtfactor(dose,step_size,current,speed=6):
    '''return the dwelltime factor (dt/dt0).
    dose in µC/cm²; step_size in nm; current in pA and speed in MHz.'''
    return speed*step_size**2*dose/current/100

def zoom_plot(z,xc,yc,WF,swf,pxs):
    '''plot z with center at (xc,yc)
        * WF: original writefield
        * swf: zoomed writefield
        * pxs: pixel size
    '''
    assert swf <= WF
    if xc > ((WF-swf)/2):
        xc = (WF-swf)/2
    if xc < ((swf-WF)/2):
        xc = (swf-WF)/2
    if yc > ((WF-swf)/2):
        yc = (WF-swf)/2
    if yc < ((swf-WF)/2):
        yc = (swf-WF)/2
    ix=f2i(yc,WF,pxs)
    iy=f2i(xc,WF,pxs)
    ni=int(swf/pxs)-1
    plt.imshow(z[ix-ni//2:ix+ni//2,iy-ni//2:iy+ni//2],origin='lower', extent=[xc-swf/2, xc+swf/2, yc-swf/2, yc+swf/2],interpolation="nearest", cmap=plt.cm.jet)
    plt.show()
    return

if __name__ == "__main__":
    print("no tets yet!")