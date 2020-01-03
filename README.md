# pecebl
Simulation for eBeam Lithography using Casino3, Python, CUDA and FFT.

This package requires a Nvidia's CUDA GPU [capable](https://developer.nvidia.com/cuda-gpus)

A third party software is needed for generating the psf data (i.e.[Casino3](http://www.gel.usherbrooke.ca/casino/)).

pecebl gives some basic pattern designer like : `dot, line, rectangle, ring, circle, move, replace, append`.

pecebl should make it easy:
 - to simulate a pattern exposure by using the FFT convolution (`pecebl.fft_ops.fft_exposure`).
 - to find the corrected dose distribution by using FFT deconvolution (`pecebl.fft_ops.fft_pec`).

# Installation
This package works only with [Anaconda](https://www.anaconda.com/distribution/?gclid=EAIaIQobChMIiaS9soHO5gIVSsDeCh3Lpwh7EAAYASAAEgKWKPD_BwE) distribution for Python
## Install the CUDA toolkit and NVIDIA driver
Download and install CUDA toolkit for your platform [here](https://developer.nvidia.com/cuda-downloads)
## Create a python's virtual environment
### with my yml file:
The easiest way to create your virtual environment is using my *environment.yml* file:

`conda env create -f environment.yml -n yourenv`

### or:
#### 1) create your virtual env with conda:

`conda create -n yourenv python=3.7 cudatoolkit pyqt pywin32`

#### 2) install dependencies:

`pip install ipython==7.8.0 jupyter numba numpy scipy sympy pandas pyqtgraph pyopengl matplotlib imageio pyculib pycuda scikit-cuda`

## install pecebl
Activate your virtual environment: `activate yourenv`

You can install in local mode using: `python setup.py install`

or using pip : `pip install .`

## check installation
check your installation with : `pecebl --show` if everything is fine you will see an exposure example's plot.

# Getting started
## Pattern designer and PSF import
### Create a pattern
Get photonic crystal `example1` centered at `(0,0)`, hole radius `48 nm`, pitch `170 nm` and stepsize `4 nm`

`from pecebl.designer import PatternDesigner as pg`

`final_pattern=pg.example1(a=170, r=48, ss=4)`

`from pecebl.utils import *`

`plt.plot(final_pattern[:,0], final_pattern[:,1], 'o', ms=1)`

`plt.axis('equal');plt.show()`

### Setup the electron beamer
We use a *Zeiss Supra40* SEM with `30 kV` and the `7.5 µm` aperture

`from pecebl.sem import supra40 as beamer`

`meb = beamer.Supra40(30)`

`meb.change_aperture(7.5)`

`meb.info()`

### Import data from Casino3 software
We use the psf file from *Casino3* simulation in `examples/data` folder: *ZEP520_1e7_30kV_100mrad_1pt*

`from pecebl.psf_import.Casino import Casino3 as cs3`

`sim=cs3('ZEP520_1e7_30kV_100mrad_1pt')`

The number of electron paths simulated in Casino3 was `1e7`.
The beam writer Raith Elphy Plus has `6 MHz` of electronic speed.
`i_y` for locating at the center of the psf and `i_z` for placing at the middle depth of the ebeam resist.
Now we can get the pre-psf data:

`pre_psf=get_pre_psf(1e7, sim, 6, meb.beam_current, i_y=3, i_z=3)`

## Exposure process
### Building the PSF data
*NP* is the number of pixels, *WF* is the writefield (nm). We can calculate the *pixel_size* then map the two columns data *pre_psf* to a 2D matrix *z_psf* of size *(WF, WF)* $(nm^2)$ (or *(NP, NP)* $(pixel^2)$):

`NP = 2048; WF = 5000`

`pixel_size=np.float32(WF/NP)`

`from pecebl.ebl_kernels import kernels as ker`

`z_psf=ker.build_psf(pre_psf, NP, WF, pixel_size, pg.dot(0,0)[0])`

### Padding
We need to transform the *z_psf* data prior to apply the FFT (Victor Podlozhnyuk white paper)

`ppsf=np.empty((NP,NP),np.float64)`

`ppsf[:NP//2-1,NP//2+1:]=z_psf[NP//2+1:,:NP//2-1]`

`ppsf[:NP//2-1,:NP//2+1]=z_psf[NP//2+1:,NP//2-1:]`

`ppsf[NP//2-1:,:NP//2+1]=z_psf[:NP//2+1,NP//2-1:]`

`ppsf[NP//2-1:,NP//2+1:]=z_psf[:NP//2+1,:NP//2-1]`

`del z_psf`

### Building the dose distribution
We need to 'cut' data in blocks and grid for parallel calculation on GPU.

`from sympy.ntheory import primefactors`

`primefactors(final_pattern.shape[0])`

So we cut the *final_pattern* into grid of blocks size: `(11*61, 3*137)`

Now we can get dose distribution data: *dose_dis* is the initial dose distribution for our pattern. Default dose factor is *1* at each dot of the pattern.

`dose_dis = ker.build_dose_distribution(final_pattern, NP, WF, pixel_size, blockdim=(671,1), griddim=(411,1))`

We can change the exposure dose for $30\mu C/cm^2$ (`ss = 4`, `speed = 6`):

`dose_dis *= dtfactor(30,4,meb.beam_current,6)`

### Exposure
We have the PSF and the dose distribution, we can do a FFT convolution to expose our pattern:

`from pecebl.fft_ops import fft_ops as fft`

`z = fft.fft_exposure(ppsf, dose_dis)`

`print(np.min(z.real),np.min(z.imag),np.max(z.real),np.max(z.imag))`

`plt.imshow(z.real,origin='lower', extent=[-WF/2, WF/2, -WF/2, WF/2],interpolation="nearest", cmap=plt.cm.jet)`

`plt.show()`

## Develop
The development process is simplified by a threshold operation. We use a threshold of *3 eV* for ZEP520A ebeam resist.

`th_resist = 3`

`z_dev = (z.real> th_resist) * z.real`

`z_dev[z_dev > 0] = 1`

plot the development result:

`plt.imshow(z_dev,origin='lower', extent=[-WF/2, WF/2, -WF/2, WF/2])`

`plt.show()`
