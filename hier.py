#from curses import window
import numpy as np
from timeit import default_timer as timer
from scipy.interpolate import interp1d
import os
from core import *

import warnings
warnings.filterwarnings("ignore")


#implements a univariate sketch
class HierarchicalSketch():                                                  

    def __init__(self, min_error_thresh, blocksize, pfn, sfn, start_level):
        self.error_thresh = min_error_thresh
        # for quantization
        #self.TURBO_CODE_LOCATION = "./../Turbo-Range-Coder/turborc" 
        #self.TURBO_CODE_PARAMETER = "-20" #on my laptop run -e0 and find best solution
        self.coderange = np.ceil(1.0/(min_error_thresh*2))
        self.blocksize = blocksize #must be a power of 2
        self.d = int(np.log2(blocksize))
        self.pfn = pfn
        self.sfn = sfn
        self.start_level = start_level
    
    def quantize(self, x):
        #x = x.copy()
        x = np.rint(x*self.coderange)
        x = x.astype(np.int16)
        return x
    
    def dequantize(self, x_quant):
        #x_quant = x_quant.copy()
        x_quant = x_quant/self.coderange
        #x_quant = x_quant.astype(np.float16)
        return x_quant
    
    def findOptPartition(self, x: np.array):
        N = x.shape[0]
        # compute prefix sums P_i
        P = np.cumsum(x)
        # compute squared prefix sums R_i
        R = np.cumsum(np.square(x))

        mxs = []
        for l in range(0,N-1):
            SSE_L = (l+1)*((R[l] / (l+1)) - (P[l]/(l+1))**2)
            SSE_R = (N-l-1)*((R[N-1] - R[l])/(N-l-1) - ((P[N-1] - P[l])/(N-l-1))**2)
            mxs.append(max(SSE_L, SSE_R))
        return np.argmin(np.array(mxs))
    
    
    def mean_poool_optim(self, x, width):
        if width == 1:
            return x.copy()

        N = x.shape[0]
        P = np.cumsum(x)

        result = np.zeros((N // width))

        result[0] = P[width-1]
        for i in range(width, N, width):
            result[i//width] = P[i+width-1] - P[i-1]
        result /= width
        return result

    def pool(self, x, fn, width):
        slices = x.reshape(-1,width)
        N,_ = slices.shape
        return np.array([fn(slices[i]) for i in range(N)])

    def pool_max(self, x, width):
        return np.max(x.reshape(-1,width), axis=1)
    
    def pool_mean(self, x, width):
        return np.mean(x.reshape(-1,width), axis=1)
    
    def pool_median(self, x, width):
        return np.median(x.reshape(-1,width), axis=1)
    
    def pool_percentile(self, x, width, p):
        return np.percentile(x.reshape(-1,width),p, axis=1)
    
    def pool_midrank(self, x, width):
        return 0.5*(np.max(x.reshape(-1,width), axis=1) + np.min(x.reshape(-1,width), axis=1))

    def spline(self, p, width, inter):
        N = p.shape[0]
        
        #degenerate case
        if N == 1:
            return np.ones(N*width)*p[0]
        
        #treat every obs as midpoint of its range
        fn = interp1d(np.arange(0,N*width, width) + (width-1)/2, p, \
                      kind=inter, fill_value="extrapolate")
                      
        return fn(np.arange(0,N*width,1))

    def spline_optim(self, p, width):
        N = p.shape[0]

        #degenerate case
        if N == 1:
            return np.ones(N*width)*p[0]
        return np.repeat(p, width)
  
    #only works on univariate data
    def encode(self, data):
        cpy = data.copy()
        N = data.shape[0]
        self.nblks = int(np.rint(N / self.blocksize))
        hierarchies = []

        for j in range(self.nblks):
            curr = cpy[j*self.blocksize:(j+1)*self.blocksize]
            hierarchy = [] 
            residuals = []

            for i in range(self.start_level, self.d + 1):
                
                w = self.blocksize // 2**i
                
                #v = self.mean_poool_optim(curr, w)
                v = self.pool_mean(curr, w)
                #v = self.pool_percentile(curr, w, 90)
                #v = self.pool_median(curr, w)
                #v = self.pool_midrank(curr, w)
                v_quant = self.quantize(v)
                v = self.dequantize(v_quant)
                vp = self.spline_optim(v, w)

                curr -= vp 
                r = self.pool_max(np.abs(curr), w)
                #curr[np.repeat(r < self.error_thresh, w)] = 0
                
                hierarchy.append(v_quant)

                residuals.append(np.max(r))
            #print(list(residuals))
            hierarchies.append(list(zip(hierarchy, residuals)))
        return hierarchies

    def decode(self, sketch, error_thresh=0):
        
        #start = timer()
        W = np.zeros((len(sketch), self.blocksize)) #preallocate

        for i, (h,r) in enumerate(sketch, start = self.start_level):
            
            dims = h.shape[0]
            
            W[i-self.start_level,:] = self.spline_optim(h, self.blocksize // dims)
            
            if r < self.error_thresh:
                break
        #print('time:', timer()-start, 'error:', r)

        return self.dequantize(np.sum(W,axis=0))


    #packs all of the data into a single array
    def pack(self, sketch):
        vectors = []
        for h,r in sketch:
            vector = np.concatenate([np.array([r]), h])
            vectors.append(vector)
        return np.concatenate(vectors)

    #unpack all of the data
    def unpack(self, array, error_thresh=0):
        sketch = []
        for i in range(self.start_level,self.d+1):
            
            r = array[0]
            h = array[1:2**i+1]
            array = array[2**i+1:]
                
            sketch.append((h,r))
            
            if r < error_thresh:
                break

        return sketch


class MultivariateHierarchical(CompressionAlgorithm):

    '''
    The compression codec is initialized with a per
    attribute error threshold.
    '''
    def __init__(self, target,pfn = np.mean, error_thresh=1e-5, blocksize=4096, start_level = 0, trc = False):

        super().__init__(target, error_thresh)
        self.trc = trc
        self.blocksize = blocksize
        self.start_level =start_level
        self.TURBO_CODE_PARAMETER = "20"
        self.TURBO_CODE_LOCATION = "./../Turbo-Range-Coder/turborc" 
        #self.TURBO_CODE_PARAMETER = "-20" #on my laptop run -e0 and find best solution

        self.pfn = pfn

        self.sketch = HierarchicalSketch(self.error_thresh, blocksize, start_level = self.start_level, pfn=self.pfn, sfn='nearest')

    def compress(self):

        start = timer()
        arrays = []
        
        for j in range(self.p):
            vector = self.data[:,j].reshape(-1)
            ens = self.sketch.encode(vector)

            
            for en in ens:
                #cumulative_gap = min(self.error_thresh - en[-1][1], cumulative_gap)
                arrays.append(self.sketch.pack(en))
        

        codes = np.vstack(arrays).astype(np.float16)

        
        #fname = self.CODES

        trc_flag = '-' + self.TURBO_CODE_PARAMETER
        # flush to .npy file
        self.path = self.CODES + '.npy'
        np.save(self.path, codes.flatten(order='F'))
        self.CODES += '.rc'
        self.DATA_FILES[0] = self.CODES
        print('\n')
        
        # run TRC [compression]
        # best performing function should be the the int trc_flag after run of turborc -e0
        #subprocess.run(['./../Turbo-Range-Coder/turborc', trc_flag, self.path, self.CODES])
        command = " ".join(['./../Turbo-Range-Coder/turborc', trc_flag, self.path, self.CODES])
        os.system(command)

        self.compression_stats['compression_latency'] = timer() - start
        self.compression_stats['compressed_size'] = self.getSize()
        self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
        #self.compression_stats.update(struct.additional_stats)


    def decompress(self, original=None, error_thresh=1e-4):
        start = timer()
        
        #subprocess.run(['./../Turbo-Range-Coder/turborc', '-d', self.CODES, self.path])
        command = " ".join([self.TURBO_CODE_LOCATION, "-d", self.CODES, self.path])
        os.system(command)
        
        packed = np.load(self.path)
        print('\n')

        #unpack_time = timer() - start
        #print('trc time: ', unpack_time)
        packed = packed.reshape(self.p*self.sketch.nblks, -1, order='F')
        
        #start = timer()


        normalization = np.load(self.NORMALIZATION)
        _, P2 = normalization.shape

        p = int(P2 - 1)
        N = int(normalization[0,p])
        codes = np.zeros((N,p))

        #normalize_time = timer() - start
        #print('normalize time: ', normalize_time)
        #start = timer()

        j = -1
        #k=0
        #index the vstacked arrs
        start = timer()
        for i in range(self.p*self.sketch.nblks):
            # detects new column
            if i % self.sketch.nblks == 0:
                # index og codes
                k = 0
                # index blocks
                j += 1
            #start = timer()
            sk = self.sketch.unpack(packed[i,:], error_thresh)
            #unpack_time = timer() - start
            #print('unpack time: ', unpack_time)

            # print('decompress end', sk)
            #start = timer()
            codes[k*self.blocksize:(k+1)*self.blocksize, j] = self.sketch.decode(sk, error_thresh)
            #decode_time = timer() - start
            #print('decode time: ', decode_time)
            k += 1

        #unpack_time = timer() - start
        #print('unpack time: ', unpack_time)
        
        #decode_time = timer() - start
        #print('decode time: ', decode_time)

        #start = timer()

        for i in range(p):
            codes[:,i] = (codes[:,i])*(normalization[0,i] - normalization[1,i]) + normalization[1,i]


        #denormalize_time = timer() - start
        #print('denormalize time: ', denormalize_time)

        self.compression_stats['decompression_latency'] = timer() - start
        #self.compression_stats['decompression_ratio'] = (codes.size * codes.itemsize)/self.compression_stats['original_size']
        if not original is None:
            #print(original-codes)
            self.compression_stats['errors'] = self.verify(original, codes)

        return codes

def bisect(x):
    N = x.shape[0]
    return x[N // 2]

####
"""
Test code here
"""
####

## Simple data
# data = np.array([[1.23, -3.5, 2.6, 15], [7.6, -10, 2.43, -7]]).T
# data = np.array([[1.23, -3.5, 2.6, 15, 20, 13, 9, -17], [7.6, -10, 2.43, -7, 4.2, 7.9, 15.2, -17]]).T

##Bruno
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data_old/traffic.npy')[:4096*64,:]
data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/sensor_ht.npy')[:4096*128,]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/bitcoin.npy', allow_pickle = True)[:4096*64,:]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/power.npy')[:4096*256,:]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/watch.npy')[:4096*64,-1:]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/phones_accelerometer.npy')[:4096*64,1:2]

## Gabe
#data = np.load('/Users/gabemersy/Desktop/data/sensor.npy')[:4096*144,1:]
#data = np.load('/Users/gabemersy/Desktop/data/gas.npy')[:4096*36,1:]

##Sanjay
#data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:1024,1:2]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
""""""
##Test
#N,p = data.shape

#print(data.shape)
"""
New parameter guide:

* quantization + trc:      trc = True, quant = True
* quantization + gzip:     trc = False, quant = True (untested)
* Stavros method + gzip:   trc = False, quant = False
* Stavros method + trc:    trc = True, quant = False (untested)

- ensure that error_thresh >= 0.001 (1e-3), as there might be rounding errors with anything less
- larger block sizes tend to yield better compression ratios (confirm empirically)

"""

nn = MultivariateHierarchical('hier', error_thresh = 0.001, blocksize=4096*4, start_level = 11, trc = True)
nn.load(data)
nn.compress()
# note that the decoding error can be defined at decompression time
# we would do so by specifying the parameter error_thresh = 0.1 for instance
nn.decompress(data)
print(nn.compression_stats)



