from unittest import TestCase

from pecebl.designer import designer as pg
from pecebl.psf_import.casino import Casino3 as cs3
from pecebl.ebl_kernels import kernels as ker
from pecebl.sem import supra40 as beamer
from pecebl.utils import *

import os
import imageio
import ctypes

class Testpecebl(TestCase):
    def test_fft_expo(self):
        #1) create photonic crystal pattern (phC):
        r, a, ss = 48, 170, 4
        NP, WF = 2048, 5000

        start = timer()
        final_pattern = pg.example1(a,r,ss)
        dt = timer() - start
        print("total {} points created in {} s".format(final_pattern.shape[0], dt))
        playsound(freq,duration)

        #2) SEM setup and import simulated data from Casino3
        meb = beamer.Supra40(30) #we use a sem Zeiss Supra40 at 30kV
        meb.change_aperture(7.5) #use the 7.5 µm aperture
        print(meb.info())

        #casino_file='./examples/data/zep520_1e7_30kV_100mrad_1pt' #data from Casino3 software
        casino_file = args.filename
        sim=cs3(casino_file)

        # the data is simulated with 1e7 electrons
        # the ElphyPlus has 6e6 Hz of speed
        # iy, i_y  = 3, 3 : we are at the peak of the psf and at the mid-depth of the ebeam resist
        # according to the simulation parameters
        # we prepare the psf:
        pre_psf=get_pre_psf(1e7, sim, 6, meb.beam_current, i_y=3, i_z=3)

        #3) Exposure simulation
        #3-1) calculate pixel size
        pixel_centered=True #True: (0,0) is in the [NP/2,NP/2] location.
        pixel_size=np.float32(WF/NP) if pixel_centered else np.float32(WF/(NP-1))

        #3-2) build the psf
        z_psf = ker.build_psf(pre_psf, NP, WF, pixel_size, pg.dot(0,0)[0], blockdim=(32,16))

        #3-3) padded psf before FFT:
        ppsf=np.empty((NP,NP),np.float64)
        ppsf[:NP//2-1,NP//2+1:]=z_psf[NP//2+1:,:NP//2-1]
        ppsf[:NP//2-1,:NP//2+1]=z_psf[NP//2+1:,NP//2-1:]
        ppsf[NP//2-1:,:NP//2+1]=z_psf[:NP//2+1,NP//2-1:]
        ppsf[NP//2-1:,NP//2+1:]=z_psf[:NP//2+1,:NP//2-1]
        del z_psf

        #3-4) build dose distribution:
        dose_dis = ker.build_dose_distribution(final_pattern, NP, WF, pixel_size, blockdim=(671,1), griddim=(411,1))

        #3-4) Let's exposure at 30 µC/cm²: use FFT convolutiuon
        dose_dis *= dtfactor(30,ss,meb.beam_current)

        from pecebl.fft_ops import fft_ops as fft

        z = fft.fft_exposure(ppsf, dose_dis)

        #Develop
        th_resist=3 #clearing dose = 3 eV
        z_dev = (z.real> th_resist) * z.real

        z_dev[z_dev > 0] = 1

        #plt.imshow(z_dev,origin='lower', extent=[-WF/2, WF/2, -WF/2, WF/2])
        #plt.show()

        print("\nTEST PASSED.")