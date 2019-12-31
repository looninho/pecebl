from ..utils import timer
import numpy as np

import pycuda.autoinit
import pycuda.driver as cudadriver
import pycuda.gpuarray as gpuarray

#for using FFT with scikitcuda:
import skcuda.fft as cu_fft
import skcuda.linalg as linalg

def fft_exposure(ppsf, dose_dis):
    '''ebl exposure by FFT convolution of the padded PSF and the dose distribution.
        * ppsf (array): (NP, NP) of padded PSF
        * dose_dis (array): (NP, NP) of dose distribution
        
        return the ebl exposition.
    '''
    start = timer()
    #get [ TF(psf) ] and TF[ dose_dis]
    print('GPU : FFT(psf) = psf in-place fft..')
    tx = np.asarray(ppsf, np.complex128)
    fftpsf_gpu = gpuarray.to_gpu(tx)
    plan = cu_fft.Plan(fftpsf_gpu.shape, np.complex128, np.complex128)
    cu_fft.fft(fftpsf_gpu, fftpsf_gpu, plan)

    print('GPU : FFT(dose_pc) = dose_dis in-place fft..')
    tx = np.asarray(dose_dis, np.complex128)
    out_gpu = gpuarray.to_gpu(tx)
    plan = cu_fft.Plan(out_gpu.shape, np.complex128, np.complex128)
    cu_fft.fft(out_gpu, out_gpu, plan)

    #perform point-wise multiplication
    print('GPU : point-wise multiplication of two matrices...')
    linalg.init()
    out_gpu = linalg.multiply(out_gpu, fftpsf_gpu)

    #free fftpsf_gpu:
    fftpsf_gpu.gpudata.free()

    #apply inverse FFT
    print('GPU : inverse FFT to get exposure energy distribution...')
    plan = cu_fft.Plan(out_gpu.shape, np.complex128, np.complex128)
    cu_fft.ifft(out_gpu, out_gpu, plan, True)
    z=out_gpu.get()

    dt = timer() - start
    print("convolution done on GPU in %f s" % dt)
    #winsound.Beep(freq, duration)

    free, total = cudadriver.mem_get_info()
    print ('%.1f %% of device memory is free.' % ((free/float(total))*100))

    out_gpu.gpudata.free()
    free, total = cudadriver.mem_get_info()
    print ('%.1f %% of device memory is free.' % ((free/float(total))*100))
    
    return z

def fft_pec(ppsf, z_target):
    '''proximity effect correction by FFT deconvolution of the padded PSF and the dose 
        distribution.
        * ppsf (array): (NP, NP) of padded PSF
        * z_target (array): (NP, NP) the desired exposure
        
        return the dose distribution pec.
    '''

    start = timer()
    # get 1/[FFT(psf)]
    # 1) apply FFT to the convolution kernel psf
    print('GPU : FFT(psf) = psf in-place fft..')
    tx = np.asarray(ppsf, np.complex128)
    fftpsf_gpu = gpuarray.to_gpu(tx)
    plan = cu_fft.Plan(fftpsf_gpu.shape, np.complex128, np.complex128)
    cu_fft.fft(fftpsf_gpu, fftpsf_gpu, plan)

    # 2) apply inverse Hadamard product to the previous result
    print('CPU : 1/FFT(psf)...')
    fft_psf = fftpsf_gpu.get()
    fftpsf_div=1./fft_psf

    #deconvolution to get dose distribution:
    # 3) apply FFT to the target exposure z_target
    print('GPU : FFT(T) = z_target in-place fft..')
    tx = np.asarray(z_target, np.complex128)
    fftztarget_gpu = gpuarray.to_gpu(tx)
    plan = cu_fft.Plan(fftztarget_gpu.shape, np.complex128, np.complex128)
    cu_fft.fft(fftztarget_gpu, fftztarget_gpu, plan)

    # 4) perform the point-wise multiplication of the two preceding results
    print('GPU : point-wise multiplication of two matrices...')
    fftpsf_div_gpu=gpuarray.to_gpu(fftpsf_div)
    linalg.init()
    pec_gpu = linalg.multiply(fftztarget_gpu, fftpsf_div_gpu)

    # 5) apply inverse FFT to the result of the multiplication
    print('GPU : inverse FFT to get dose distribution...')
    plan = cu_fft.Plan(pec_gpu.shape, np.complex128, np.complex128)
    cu_fft.ifft(pec_gpu, pec_gpu, plan, True)
    pec = pec_gpu.get()

    dt = timer() - start
    print("deconvolution done on GPU in %f s" % dt)
 
    # check the free memory:
    free, total = cudadriver.mem_get_info()
    print ('%.1f %% of device memory is free.' % ((free/float(total))*100))

    # free gpu memories:
    #fftpsf_gpu.gpudata.free()
    fftztarget_gpu.gpudata.free()
    fftpsf_div_gpu.gpudata.free()
    pec_gpu.gpudata.free()
    free, total = cudadriver.mem_get_info()
    print ('%.1f %% of device memory is free.' % ((free/float(total))*100))

    return pec

if __name__ == "__main__":
    print("no test yet!")