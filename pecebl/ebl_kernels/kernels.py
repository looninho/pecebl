import math
import numpy as np
from numba import cuda

from ..utils import timer

import ctypes

#for device memory manager
freeMem = ctypes.c_size_t()
totalMem = ctypes.c_size_t()
result = ctypes.c_int()

## for kernel use:
def i2f(idx,WF,px_size):
    '''It is the reverse function of float2idx.'''
    return -WF/2+idx*px_size

def f2i(x,WF,px_size):
    '''It just converts a float value to index integer in the array of WF.'''
    return int((x+WF/2)/px_size)

f2i_gpu = cuda.jit(device=True)(f2i)

@cuda.jit
def pattern2matrix(dest, pattern, WF, pxs):
    '''Map the pattern to matrix data.
    It will also correct redundance dots in pattern.'''

    #we use 2D grid of 2D block:
    blockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
    threadId = blockId * (cuda.blockDim.x * cuda.blockDim.y) \
        + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
    ix = f2i_gpu(pattern[threadId,0],WF,pxs)
    iy = f2i_gpu(pattern[threadId,1],WF,pxs)
    dest[iy,ix] += pattern[threadId,2]

def psf_interp1(r,sr,M,data):
    '''You don't need to use this function.
    It's a subroutine for kernel to interpolate points in beetwen two simulated data.'''
    r_i = r/sr
    i_lo=int(r_i);i_hi = i_lo+1 if r_i > i_lo else i_lo
    if i_hi >= M:
        return 0.0
    elif i_hi == i_lo:
        return data[i_hi,1]
    else:
        slope = (data[i_hi,1]-data[i_lo,1])/(data[i_hi,0]-data[i_lo,0])
        return slope*(r-data[i_lo,0]) + data[i_lo,1]

psf_interp1_gpu = cuda.jit(device=True)(psf_interp1)

@cuda.jit
def psf_kernel(psf,point,NP,WF,pixel_size,sr,nr,simdata):
    '''This kernel stores the PSF function in the psf 2D-array.
    point: only a dot at (x=0,y=0,dose=1) the center of the writefield (WF).
    NP: number of pixel in one dimension of WF. Here we work on a square 2D-array.
    for saving the GPU time, I think it's better to give the size of the pixel instead of letting the kernel calculate it.
    sr, nr are the step and the number of points respectivly in the electron-range direction of the simulated data, simdata.
    simdata: simulated data by Casino. its shape is two columns: one for x and one for y (the deposited energy at point x).
    '''
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    for x in range(startX, NP, gridX):
        val_x = -WF/2 + x*pixel_size
        for y in range(startY, NP, gridY):
            val_y = -WF/2 + y*pixel_size
            r = math.sqrt((val_x-point[0])**2+(val_y-point[1])**2)
            psf[y, x] += point[2]*psf_interp1_gpu(r,sr,nr,simdata)

def build_psf(mcdata, NP, WF, pixel_size, location, blockdim=(32,16)):
    '''return an array (NP,NP) of PSF
        * mcdata (array): simulated data,
        * NP (int): number of pixels of a squared output data,
        * WF (float): the writefield size in nm,
        * location (array): (x, y, dose) PSF peak coordonnates. (x,y) in nm,
        * pixel_size (float): pixel size in nm,
        * blockdim (int, int): blockdim for GPGPU calculation
    '''
    griddim = (int(NP/blockdim[0]),int(NP/blockdim[1]))
    z_psf = np.zeros((NP, NP), dtype = np.float64)
    
    start = timer()
    out_gpu = cuda.to_device(z_psf)
    in_gpu = cuda.to_device(mcdata)

    sr=mcdata[1,0]-mcdata[0,0];nr=mcdata.shape[0] #step in r and number of steps

    # call the kernel for mapping PSF to 2D-array:
    psf_kernel[griddim, blockdim](out_gpu,location,NP,WF,pixel_size,sr,nr,in_gpu) 

    # copy the result back to CPU and release GPU memory:
    out_gpu.to_host()
    in_gpu.to_host()

    #stop the timer and print the min, max values:
    dt = timer() - start
    print("z_PSF created on GPU in %f s" % dt)
    log_message = f'min value = {np.min(z_psf)}; max value = {np.max(z_psf)}; sum value = {np.sum(z_psf)}.'
    print(log_message)

    out_gpu.gpu_data._mem.free()
    in_gpu.gpu_data._mem.free()

    result = cuda.driver.driver.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
    print("  Total Memory: %ld MiB" % (totalMem.value / 1024**2))
    print("  Free Memory: %ld MiB" % (freeMem.value / 1024**2))
    
    return z_psf

def build_dose_distribution(pattern, NP, WF, pixel_size, blockdim, griddim):
    '''return an array (NP,NP) of dose distribution
        * pattern (array): the pattern, its shape is (n,3) where n is 
            the number of dots
        * NP (int): number of pixels of a squared output data,
        * WF (float): the writefield size in nm,
        * location (array): (x, y, dose) PSF peak coordonnates. (x,y) in nm,
        * pixel_size (float): pixel size in nm,
        * blockdim (int, int): block dim for GPGPU calculation,
        * griddim (int, int): grid dim for GPGPU calculation
    '''
    dose_dis = np.zeros((NP, NP), dtype = np.float64)

    start = timer()
    out_gpu = cuda.to_device(dose_dis)
    in_gpu = cuda.to_device(pattern)

    pattern2matrix[griddim, blockdim](out_gpu,in_gpu,WF,pixel_size)

    out_gpu.to_host()
    in_gpu.to_host()
    dt = timer() - start
    print("Dose calculated on GPU in %f s" % dt)
    log_message = f'overlap = {dose_dis.sum()-pattern[:,2].sum()}. Negative value = lost dose.'
    print(log_message)

    print(np.max(pattern[:,2]),np.max(dose_dis))

    out_gpu.gpu_data._mem.free()
    in_gpu.gpu_data._mem.free()
    result = cuda.driver.driver.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
    print("  Total Memory: %ld MiB" % (totalMem.value / 1024**2))
    print("  Free Memory: %ld MiB" % (freeMem.value / 1024**2))
    
    return dose_dis
    
if __name__ == "__main__":
    print("no test yet!")