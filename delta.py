import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *
from greedy_column_selection import *
from mtscompress_v2 import *
import random


class Sprintz(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/(error_thresh)))


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes, decode = delta_xform(codes)

		fname = self.target + '/delta'
		np.save(fname, decode)

		self.DATA_FILES += [fname + '.npy']

		codes = codes.astype(np.intc)

		struct = iarray_bitpacking(codes, order = 'F')
		struct.flush(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.load(self.CODES)
		codes = codes.astype(np.float64)

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		fname = self.target + '/delta.npy'
		decode = np.load(fname)
		codes = i_delta_xform(codes, decode)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


class SprintzGzip(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/(error_thresh)))


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes, decode = delta_xform(codes)

		fname = self.target + '/delta'
		np.save(fname, decode)

		codes = codes.astype(np.intc)

		struct = iarray_bitpacking(codes, order = 'F')
		struct.flushbz2(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadbz2(self.CODES)
		codes = codes.astype(np.float64)


		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		fname = self.target + '/delta'

		decode = np.load(fname+'.npy')
		codes = i_delta_xform(codes, decode)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes

class SprintzLearnedGzip(CompressionAlgorithm):
	def __init__(self, target, error_thresh=0.005, trc = False, col_sel_algo = None):
		super().__init__(target, error_thresh)
		self.trc = trc
		self.TURBO_CODE_PARAMETER = "20"
		self.col_sel_algo = col_sel_algo

		self.coderange = int(math.ceil(1.0/(error_thresh)))
	
	def getModelSize(self):
		return self.model_size

	"""The main compression loop
	"""
	def compress(self, lags = 1, param_error_thresh = 0.001):
		N, p = self.N, self.p
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		univ_lin = UnivariateTS_IM(lags)
		
		self.lin = univ_lin

		codes, self.model_size = univ_lin.compress(codes,param_error_thresh = param_error_thresh, delta = True)
		codes = codes.astype(np.intc)

		if not self.trc:
			struct = iarray_bitpacking(codes, order='F')
			struct.flushz(self.CODES)
			struct.flushmeta(self.METADATA)
		else:
			trc_flag = '-' + self.TURBO_CODE_PARAMETER
			# flush to .npy file
			self.path = self.CODES + '.npy'
			np.save(self.path, codes.flatten(order='F'))
			#np.save(self.path, codes)

			self.CODES += '.rc'
			self.DATA_FILES[0] = self.CODES
			print('\n')
			
			# run TRC [compression]
			# best performing function should be the the int trc_flag after run of turborc -e0
			subprocess.run(['./../Turbo-Range-Coder/turborc', trc_flag, self.path, self.CODES])

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize() + self.getModelSize()
		self.compression_stats['compressed_ratio'] = (self.getSize()+ self.getModelSize())/self.compression_stats['original_size']

		

	def decompress(self, original=None):
		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadz(self.CODES)

		codes = self.lin.decompress(codes)

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		for i in range(p):
			codes[:,i] = (codes[:,i]/self.coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes

####
"""
Test code here
"""
####

"""
data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:,1:]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
#normalize this data
N,p = data.shape
nn = SprintzGzip('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/trios/sigma2_10.npy')
# data = np.take(np.load('/Users/gabemersy/Desktop/data/house.npy'), [0, 1, 3], axis=1)
# data = np.load('/Users/gabemersy/Desktop/data/gas.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/trios/sigma2_4.npy')
# data = np.load('/Users/gabemersy/Desktop/data/electricity.npy')[:, 0:10]
# data = np.take(np.load('/Users/gabemersy/Desktop/data/solar.npy'), [1, 2, 3, 4, 6, 7], axis=1)
# data = np.take(np.load('/Users/gabemersy/Desktop/data/taxi.npy'), [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18], axis = 1)
# data = np.load('/Users/gabemersy/Desktop/data/taxi.npy')
# data = np.load('/Users/gabemersy/Desktop/data/gas.npy')
# data = np.load('/Users/gabemersy/Desktop/data/house.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/trios/sigma2_10.npy')
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/phones_accelerometer.npy')[:4096*64,]

"""
N, p = data.shape
# nn = SprintzLearnedGzip('quantize', trc = False, error_thresh = 0.005)
nn = SprintzGzip('quantize', error_thresh = 0.001)
nn.load(data)
nn.compress()
# # nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""