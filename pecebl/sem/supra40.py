import numpy as np

## max probe current in pA on Zeiss Supra 40 SEM
MAX_CURRENT_APERTURE_30KV = 7300
MAX_CURRENT_APERTURE_20KV = 4340
MAX_CURRENT_APERTURE_10KV = 2750
MAX_CURRENT_APERTURE_5KV = 2000

class Supra40:    
    def __init__(self,kV=30, maxpc=None, i=-1):
        '''class for sem.
            you have to initialize your instance with:
            Args:
                * kV: the working voltage in (kV)
                * maxpc (pA): the current for the biggest aperture.
                * i (int): the ith aperture that you have measured the maxpc
                    i=0 is the smallest aperture 7.5 µm.
                    default (3) is 30 µm aperture.
        '''
        self._set_apertures()
        self.current_aperture_idx = i
        self.change_kV(kV, maxpc,i)
    
    def info(self):
        print("SEM info:")
        print("current aperture size: {} µm".format(self.aperture))
        print("current voltage: {} kV".format(self.kV))
        print("beam current: {} pA\n".format(self.beam_current))
    
    def change_kV(self, kV, maxpc=None, i=-1):
        '''change the current voltage.
            Args:
                * kV : the working voltage in (kV)
                * maxpc (pA): the current for the biggest aperture.
                * i (int): the ith aperture that you have measured the maxpc
                    i=0 is the smallest aperture 7.5 µm.
        '''
        assert i < len(self.apertures)
        self.kV = kV
        if maxpc == None:
            if kV == 30:
                self.max_probe = MAX_CURRENT_APERTURE_30KV
            elif kV == 20:
                self.max_probe = MAX_CURRENT_APERTURE_20KV
            elif kV == 10:
                self.max_probe = MAX_CURRENT_APERTURE_10KV
            elif kV == 5:
                self.max_probe = MAX_CURRENT_APERTURE_5KV
            else:
                print("ERROR: this voltage {} kV is not in [5, 10, 20, 30] kV.\n \
                 maxpc argument missed!".format(kV))
                return
        else:
            if i == -1 or i == len(self.apertures) -1:
                self.max_probe = maxpc
            else:
                self.max_probe = self._get_pc2(maxpc, i)[-1]    
        return self.change_aperture(self.apertures[self.current_aperture_idx])

    def change_aperture(self, aperture_size):
        '''change the working aperture size.
        Arg:
            * aperture_size (µm): diameter of aperture.
            Default Value : 7.5,10,20,30,60 or 120
            return the beam current (pA) for this aperture size.
        '''
        assert aperture_size in self.apertures
        self.current_aperture_idx = self.apertures.index(aperture_size)
        self.aperture = self.apertures[self.current_aperture_idx]
        self.beam_current = self._get_pc(self.max_probe)[self.current_aperture_idx]
        return self.beam_current

    def _set_apertures(self, apertures=[7.5,10,20,30,60,120]):
        '''set the aperture sizes.
        Args:
            * apertures : list of aperture diameters (µm).
        return the number of apertures for this sem.
        '''
        self.apertures = apertures
        return len(self.apertures)

    # below these two functions are for a typical SEM Zeiss Supra 40:
    def _get_pc(self, maxpc):
        ''' get probe currents for all apertures.
        Args:
            * maxpc : beam current at biggest aperture (pA).
        example : get_pc(7300) return an array of beam currents for different 
        aperture sizes at 30 kV (7300 pA).'''
        im = len(self.apertures)-2
        pc=np.array([maxpc]*(im+2))
        for ii in np.arange(im,-1,-1):
            pc[ii] = pc[ii+1]*(self.apertures[ii]/self.apertures[ii+1])**2
        return pc

    def _get_pc2(self, pc_i,i):
        '''get probe currents for all apertures.
        Args:
            * pc_i : beam current (pA) at ith aperture. i=0 is the smallest aperture.
            * i (int): the ith aperture that you measure pci
        example : get_pc2(30,456,3) return an array of beam currents for different
        aperture sizes at 30 kV.
        '''
        assert i >= 0 and i < len(self.apertures)
        pc=np.array([0]*len(self.apertures))
        pc[i] = pc_i
        pc[-1] = pc[i]*(self.apertures[-1]/self.apertures[i])**2
        for ii2 in np.arange(len(self.apertures)-2,-1,-1):
            pc[ii2] = pc[ii2+1]*(self.apertures[ii2]/self.apertures[ii2+1])**2
        return pc

if __name__ == "__main__":
    sem = Supra40(30)
    sem.info()
    sem.change_aperture(7.5)
    sem.info()
    sem.change_kV(20)
    sem.info()
