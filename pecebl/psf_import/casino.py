# -*- coding: utf-8 -*-
"""
Class for import simulated data from Casino software:
https://www.gel.usherbrooke.ca/casino/index.html

class Casino2 for data from CASINO v2.xx
class Casino3 for data from CASINO v3.x
"""

import numpy as np
from scipy import array

class Casino2:
    def __init__(self,simfilename):
        with open(simfilename,'r') as f:
            self.maxED=self.getMaxED(f)
            self.sx,self.nx=self.getDivisionSize(f,3)
            self.sy,self.ny=self.getDivisionSize(f,1)
            self.sz,self.nz=self.getDivisionSize(f,1)
            self.xrange=self.getRange(f)
            self.yrange=self.getRange(f)
            self.zrange=self.getRange(f)
            self.EDD=self.getEDD(f)
    def getMaxED(self,f):
        l=f.readline();l=f.readline()
        return float(l.split(':')[1])
    def getDivisionSize(self,f,n=1):
        for i in range(n):
            l=f.readline()
        s=float(l.split(':')[1].split('nm')[0])
        divisions=int(l.split('for')[1].split('divisions')[0])
        return s, divisions
    def getRange(self,f):
        l=f.readline()
        de=float(l.split('From:')[1].split('To:')[0])
        a=float(l.split('To:')[-1].split('nm')[0])
        return (de,a)
    def getEDD(self,f):
        #X,Y,Z = np.mgrid[self.xrange[0]:self.xrange[1]:self.sx,self.yrange[0]:self.yrange[1]:self.sy,self.zrange[0]:self.zrange[1]:slef.sz]
        V=np.zeros_like(np.ndarray((self.nx,self.ny,self.nz)))
        for iplanes in range(2): #[XZ] et [XY]
            if iplanes > 0:
                l=f.readline() # ligne vide
            l=f.readline();planename=l.split(' ')[0]
            if planename == 'XZ':
                nplanes = self.ny;i_name=1 #flag for id of y (=1), z (=2)
            else:
                nplanes = self.nz;i_name=2
            for iy_or_iz in range(nplanes): #ny for [XZ], nz for (XY]
                l=f.readline()
                i_yorz = int(l.split(' plane ')[1].split('\t')[0]) # same as iy_or_iz
                y_or_z = float(l.split('Position:')[1].split('nm')[0]) #value of y or z
                l=f.readline() # nx, values of X (de planename[0])
                l2=l.split('\t')[1:]
                X =[float(elm.split('nm')[0]) for elm in l2]
                i_zory=0;nmax = self.nz if planename[1] == 'Z' else self.ny
                while 1 and i_zory < nmax: # nz or ny, for values of planename[1]
                    try:
                        l=f.readline();l2=l.split('\t')
                        z_or_y = float(l2[0].split('nm')[0]) #value of z or y
                        l2=list(map(float,l2[1:])) #EDD values@planename[1]=f(planename[0])
                        for ix in range(len(l2)):
                            if i_name == 1:
                                V[ix,i_yorz,i_zory]=l2[ix]
                            else:
                                V[ix,i_zory,i_yorz]=l2[ix]
                        i_zory += 1
                    except EOFError:
                        break
        return V

class Casino3:
    def __init__(self,simfilename):
        with open(simfilename,'r') as f:
            self.title, self.dType=self.getDistributionType(f)
            self.maxED=self.getMaxED(f)
            #get Division Sizes:
            if self.dType=='Cylindric':
                self.sr,self.nr=self.getDivisionSize(f,3) #line of interest at 3rd line (void, Division Size, LOI)
                self.sz,self.nz=self.getDivisionSize(f,1)
                self.xc,self.yc=self.getCenter(f)
                self.zrange=self.getZRange(f,self.sz)
                self.rrange=self.getRRange(f)
                self.zlabels,self.data=self.getEDD_cyl(f,self.nz)
            elif self.dType=='Cartesian':
                self.sx,self.nx=self.getDivisionSize(f,3)
                self.sy,self.ny=self.getDivisionSize(f,1)
                self.sz,self.nz=self.getDivisionSize(f,1)
                self.xrange=self.getRange(f)
                self.yrange=self.getRange(f)
                self.zrange=self.getRange(f)
                self.data=self.getEDD_cart(f)
            else: #Spheric
                self.sr,self.nr=self.getDivisionSize(f,3) #line of interest at 3rd line (void, Division Size, LOI)
                self.xc,self.yc,self.zc=self.getCenter(f)
                self.zlabel = 'z='+str(self.zc)+'nm'
                self.data=self.getEDD_sph(f)
                return
    def getDistributionType(self,f):
        title=f.readline(); #read the first line = title of the plot
        l=f.readline()
        return title, l.split(':')[1].split(' ')[1].split('\n')[0]
    def getMaxED(self,f):
        l=f.readline()
        return float(l.split(':')[1])
    def getDivisionSize(self,f,n=1):
        for i in range(n):
            l=f.readline()
        s=float(l.split(':')[1].split('nm')[0])
        divisions=int(l.split('for')[1].split('divisions.')[0])
        return s, divisions
    def getCenter(self,f):
        line=f.readline()
        l=line.split('=')
        xc=float(l[1].split('y')[0])
        if len(l)>3:
            yc=float(l[2].split('z')[0])
            zc=float(l[-1])
            return xc,yc,zc
        else:
            yc=float(l[-1])
            return xc, yc
    def getRange(self,f):
        l=f.readline()
        de=float(l.split('From:')[1].split('To:')[0])
        a=float(l.split('To:')[-1].split('nm')[0])
        return (de,a)
    def getZRange(self,f,step):
        l=f.readline()
        de=float(l.split(':')[1].split(' ')[0])
        a=float(l.split(':')[-1].split('nm')[0])
        return np.arange(de,a,step)
    def getRRange(self,f):
        l=f.readline()
        l2=l.split('Z\\Radius\t')[1].split('nm\n')[0].split('nm\t')
        return array(list(map(float, l2)))
    def getEDD_cart(self,f):  #Cartesian
        #X,Y,Z = np.mgrid[self.xrange[0]:self.xrange[1]:self.sx,self.yrange[0]:self.yrange[1]:self.sy,self.zrange[0]:self.zrange[1]:slef.sz]
        V=np.zeros_like(np.ndarray((self.nx,self.ny,self.nz)))
        for iplanes in range(2): #[XZ] et [XY]
            if iplanes > 0:
                l=f.readline() # blank line
            l=f.readline();planename=l.split(' ')[0]
            if planename == 'XZ':
                nplanes = self.ny;i_name=1 #flag for id of y (=1), z (=2)
            else:
                nplanes = self.nz;i_name=2
            for iy_or_iz in range(nplanes): #ny for [XZ], nz for (XY]
                l=f.readline()
                i_yorz = int(l.split(' plane ')[1].split('\t')[0]) # same as iy_or_iz
                y_or_z = float(l.split('Position:')[1].split('nm')[0]) #value of y or z
                l=f.readline() # nx, values of X (for planename[0])
                l2=l.split('\t')[1:]
                X =[float(elm.split('nm')[0]) for elm in l2]
                i_zory=0;nmax = self.nz if planename[1] == 'Z' else self.ny
                while 1 and i_zory < nmax: # nz or ny, for values of planename[1]
                    try:
                        l=f.readline();l2=l.split('\t')
                        z_or_y = float(l2[0].split('nm')[0]) #value of z or y
                        l2=list(map(float,l2[1:])) #EDD values@planename[1]=f(planename[0])
                        for ix in range(len(l2)):
                            if i_name == 1:
                                V[ix,i_yorz,i_zory]=l2[ix]
                            else:
                                V[ix,i_zory,i_yorz]=l2[ix]
                        i_zory += 1
                    except EOFError:
                        break
        return V
    def getEDD_cyl(self,f,nmax): #Cylindric
        ret=[];zlabels=[];i=0
        while 1 and i < nmax:
            try:
                l=f.readline();l2=l.split('\t')
                zlabels.append(l2[0]);l2=list(map(float,l2[1:]));
                ret.append(l2);i += 1
            except EOFError:
                break
        return (zlabels,array(ret))
    def getEDD_sph(self,f):
        ret=np.zeros((self.nr, 2), dtype = np.float32)
        i=0
        l=f.readline() #just read the line "Radius"
        while 1 and i < self.nr:
            try:
                l=f.readline()
                ret[i]=[float(l.split('nm')[0]),float(l.split('\t')[1])]
                i += 1
            except EOFError:
                print(l)
                break
        return ret

if __name__ == "__main__":
    print("no test yet!")