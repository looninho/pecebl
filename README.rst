
pecebl
======

Simulation for eBeam Lithography using Casino3, Python, CUDA and FFT.

This package requires a Nvidia's CUDA GPU `capable <https://developer.nvidia.com/cuda-gpus>`_

A third party software is needed for generating the psf data (i.e.\ `Casino3 <http://www.gel.usherbrooke.ca/casino/>`_\ ).

**pecebl** gives some basic pattern designer like : ``dot, line, rectangle, ring, circle, move, replace, append``.

**pecebl** should make it easy:


* to simulate a pattern exposure by using the FFT convolution (\ ``pecebl.fft_ops.fft_exposure``\ ).
* to find the corrected dose distribution by using FFT deconvolution (\ ``pecebl.fft_ops.fft_pec``\ ).

Installation
============

This package requires `Anaconda <https://www.anaconda.com/distribution/?gclid=EAIaIQobChMIiaS9soHO5gIVSsDeCh3Lpwh7EAAYASAAEgKWKPD_BwE>`_ distribution for Python

Install the CUDA toolkit and NVIDIA driver
------------------------------------------

If not done, download and install CUDA toolkit for your platform `here <https://developer.nvidia.com/cuda-downloads>`_

Create a python's virtual environment
-------------------------------------

with my yml file:
^^^^^^^^^^^^^^^^^

The easiest way to create your virtual environment is using my *environment.yml* file:

``conda env create -f environment.yml -n youreblenv``

or if you want to create it by yourself:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``conda create -n youreblenv python=3.7 cudatoolkit pyqt``

install pecebl
--------------

Activate your virtual environment: ``activate youreblenv``

Now you can install **pecebl** in local mode by cd to your local pecebl directory then enter: ``python setup.py install``

or using pip : ``pip install pecebl``

check installation
==================

check your installation when you are in the pecebl root directory with : ``pecebl --show`` if everything is fine you will see an exposure example's plot.

Getting started
===============

I) Building the PSF data
------------------------

We will get at the end of this section a 2D matrix data with the psf at the center. Here are the steps to do:


#. Decide the hardware parameters you want to use: the beam energy, the beam current. And the physical properties of your sample.
#. Get the interaction between the electron beam and your sample. You can do it by experiment or by monte-carlo simulation like `Casino3 <http://www.gel.usherbrooke.ca/casino/>`_. We call it the *psf function*.
#. Map the *psf function* to a 2D matrix of size equals to the writefield you want to simulate. We call it the *PSF data*.
   ### I-1) Setup the electron beamer
   We use a *Zeiss Supra40* SEM with ``30 kV`` and the ``7.5 µm`` aperture

``from pecebl.sem import supra40 as beamer``

``meb = beamer.Supra40(30)``

``meb.change_aperture(7.5)``

``meb.info()``

I-2) Import data from Casino3 simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the psf file from *Casino3* simulation in ``examples/data`` folder: *zep520_1e7_30kV_100mrad_1pt*

``from pecebl.psf_import.casino import Casino3 as cs3``

``sim=cs3('zep520_1e7_30kV_100mrad_1pt')``

The number of electron paths simulated in Casino3 was ``1e7``. The beam writer Raith Elphy Plus has ``6 MHz`` of electronic speed.
``i_y`` for locating at the peak of the psf and ``i_z`` for placing at the middle depth of the ebeam resist. In this example, I use **Casino3** in a grid size of ``(x=8000, y=0.6, z=310)`` in *nm* divided by ``(nx=8000, ny=6, nz=6)`` dots, hence ``i_y=3`` and ``i_z=3``. Now we can get the ``psf_fct``\ :

``from pecebl.utils import *``

``psf_fct=get_psf_fct(1e7, sim, 6, meb.beam_current, i_y=3, i_z=3)``

I-3) Building the PSF data
^^^^^^^^^^^^^^^^^^^^^^^^^^

``NP`` is the number of pixels, ``WF`` is the writefield *(nm)*. We can calculate the ``pixel_size`` then map the two columns data ``psf_fct`` to a 2D matrix ``z_psf`` of size *(WF, WF)* $(nm^2)$ (or *(NP, NP)* $(pixel^2)$):

``NP = 2048; WF = 5000``

``pixel_size=np.float32(WF/NP)``

``from pecebl.ebl_kernels import kernels as ker``

``from pecebl.designer import designer as pg``

``z_psf=ker.build_psf(psf_fct, NP, WF, pixel_size, pg.dot(0,0)[0])``

II) Pattern designer
--------------------

II-1) Create a pattern
^^^^^^^^^^^^^^^^^^^^^^

Get photonic crystal ``example1`` centered at ``(0,0)``\ , hole radius ``48 nm``\ , pitch ``170 nm`` and stepsize ``4 nm``

``final_pattern=pg.example1(a=170, r=48, ss=4)``

``plt.plot(final_pattern[:,0], final_pattern[:,1], 'o', ms=1)``

``plt.axis('equal');plt.show()``

Building the dose distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to 'cut' data in blocks and grid for parallel calculation on GPU.

``from sympy.ntheory import primefactors``

``primefactors(final_pattern.shape[0])``

So we cut the ``final_pattern`` into grid of blocks size: ``(11*61, 3*137)``

Now we can get dose distribution data: ``dose_dis`` is the initial dose distribution for our pattern. Default dose factor is ``1`` at each dot of the pattern.

``dose_dis = ker.build_dose_distribution(final_pattern, NP, WF, pixel_size, blockdim=(671,1), griddim=(411,1))``

We can change the exposure dose for $30\mu C/cm^2$ (\ ``ss = 4``\ , ``speed = 6``\ ) by multiply a dwelltime factor:

``dose_dis *= dtfactor(30,4,meb.beam_current,6)``

III) Exposure process
---------------------

III-1) Padding the PSF data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before applying the *FFT* transformations, we need to transform the *z_psf* data (Victor Podlozhnyuk white paper)

``ppsf=np.empty((NP,NP),np.float64)``

``ppsf[:NP//2-1,NP//2+1:]=z_psf[NP//2+1:,:NP//2-1]``

``ppsf[:NP//2-1,:NP//2+1]=z_psf[NP//2+1:,NP//2-1:]``

``ppsf[NP//2-1:,:NP//2+1]=z_psf[:NP//2+1,NP//2-1:]``

``ppsf[NP//2-1:,NP//2+1:]=z_psf[:NP//2+1,:NP//2-1]``

``del z_psf``

III-2) Exposure
^^^^^^^^^^^^^^^

We have the PSF and the dose distribution, we can do a FFT convolution to expose our pattern:

``from pecebl.fft_ops import fft_ops as fft``

``z = fft.fft_exposure(ppsf, dose_dis)``

``print(np.min(z.real),np.min(z.imag),np.max(z.real),np.max(z.imag))``

``plt.imshow(z.real,origin='lower', extent=[-WF/2, WF/2, -WF/2, WF/2],interpolation="nearest", cmap=plt.cm.jet)``

``plt.show()``

IV) Develop
-----------

The development process is simplified by a threshold operation. We use a threshold of ``3 eV`` for ZEP520A ebeam resist.

``th_resist = 3``

``z_dev = (z.real> th_resist) * z.real``

``z_dev[z_dev > 0] = 1``

plot the development result:

``plt.imshow(z_dev,origin='lower', extent=[-WF/2, WF/2, -WF/2, WF/2])``

``plt.show()``

PEC
===

 In this section, we want to find the dose distribution matrix and we know the target exposure. The way to get this target exposure will be discussed later.
We start from previous section I) to get the ``z_psf`` and also its padded ``ppsf``

I) Import target exposure
-------------------------

The example is in the filename *target_ebl_for_pec.npy*

``import zipfile``

``zfile = zipfile.ZipFile("target_ebl_for_pec.zip","r")``

``with zfile as zip_ref:``
    ``zip_ref.extractall()``

``z_target=np.load(zfile.namelist()[-1])``

``plt.imshow(z_target,origin='upper', extent=[-WF/2, WF/2, -WF/2, WF/2],interpolation="nearest", cmap=plt.cm.jet)``

``plt.show()``

II) Get PEC by deconvolution
----------------------------

``pec = fft.fft_pec(ppsf,z_target)``

plotting:

``plt.imshow(pec.real,origin='upper', extent=[-WF/2, WF/2, -WF/2, WF/2],interpolation="nearest", cmap=plt.cm.jet)``

``plt.show()``

The ``pec`` found by FFT deconvolution may contain negative values, with a simple operation we can avoid it. Depend on your hardware constraint you could make some adjustment then implement the resulting dose distribution to your hardware to obtain the desired exposure.
