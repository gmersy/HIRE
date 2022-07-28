import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from core import *
import gorillacompression as gc


class Gorilla(CompressionAlgorithm):
	'''Applies a rolling xor operation like Gorilla
	'''

	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)



	"""The main compression loop
	"""
	def compress_old(self):
		start = timer()

		codes = self.data.copy() #store a copy of the dataset

		placeholder = codes.copy()
		

		for j in range(self.p):
			for i in range(1, self.N):
				codes[i, j] = self._float_xor(placeholder[i-1,j],codes[i,j])
		

		codes = codes.astype(np.float32).flatten(order='F')

		fname = self.CODES
		np.save(fname, codes)

		compressz(self.CODES + '.npy', self.CODES+'.npyz')
		self.DATA_FILES += [self.CODES + '.npyz']


		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
	
	def compress(self):
		start = timer()
		codes = self.data.copy()
		#print(codes)

		encodings = []
		bytestrings = []
		for j in range(self.p):
			content = gc.ValuesEncoder.encode_all(codes[:, j].tolist())
			encodings.append(content)
			bytestrings.append(content['encoded'])
		
		self.encodings = encodings

		
		full_bytestring = b''.join(bytestrings)		
		
		with open(self.CODES + ".bin", "wb") as f:
			f.write(full_bytestring)
			f.close() 
		compressz(self.CODES + ".bin", self.CODES + ".binz")
		self.DATA_FILES.append(self.CODES + ".binz")

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
	
	def decompress(self, original=None):
		start = timer()
		print(len(self.encodings))
		decoded = []
		for cnt, encoding in enumerate(self.encodings):
			decoded.append(gc.ValuesDecoder.decode_all(encoding))
		
		codes = np.column_stack(decoded)


		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		for j in range(p):
			codes[:,j] = (codes[:,j])*(normalization[0,j] - normalization[1,j]) + normalization[1,j]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)


		return codes

		



	def decompress_old(self, original=None):

		start = timer()

		decompressz(self.CODES + '.npyz', self.CODES+'.npy')
		codes = np.load(self.CODES+".npy")

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		codes = codes.reshape(N,p, order='F').astype(np.float64)
		coderange = np.max(codes)

		
		for i in range(1, self.N):
			for j in range(p):
				codes[i,j] = self._float_xor(codes[i-1,j],codes[i,j])
	
		for j in range(p):
			codes[:,j] = (codes[:,j])*(normalization[0,j] - normalization[1,j]) + normalization[1,j]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)


		return codes


	#zero out as many bits as possible to hit error threshold
	def _strip_code(self, vector):
		p = vector.shape[0]

		for i in range(p): #go component by component
			value = vector[i]
			ba = bytearray(struct.pack("d", value))

			for j in range(len(ba)):
				tmp = ba[j]
				ba[j] = int('00000000')
				newvalue = struct.unpack("d", ba)[0]

				if np.abs(newvalue - value) > self.error_thresh:
					ba[j] = tmp
					vector[i] = struct.unpack("d", ba)[0]
					break

		return vector, j


		#zero out as many bits as possible to hit error threshold
	def _float_xor(self, v1, v2):
		ba1 = bytearray(struct.pack("d", v1))
		ba2 = bytearray(struct.pack("d", v2))

		for j in range(len(ba1)):
			ba2[j] = ba1[j] ^ ba2[j]

		return struct.unpack("d", ba2)[0]
	# def _xor_float(self, f1, f2):
	# 	# print(f1, f2)
		
	# 	# print(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f1)))
	# 	# print('\n')
	# 	# print(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f2)))
	# 	xor = lambda x,y:(x.view("i")^y.view("i")).view("f")
	# 	return xor (np.array(f1), np.array(f1))


class GorillaLossy(CompressionAlgorithm):
	'''Applies a rolling xor operation like Gorilla
	'''

	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = self.data.copy() #store a copy of the dataset

		stripped_bits = 8

		for i in range(self.N):
			v,j = self._strip_code(codes[i,:]) #zero out as many as possible
			stripped_bits = min(stripped_bits,j) #find the minimum stripped bits


		#set the right bitsize
		if stripped_bits >= 6:
			codes = codes.astype(np.half)
			placeholder = codes.copy()
		elif stripped_bits >= 4:
			codes = codes.astype(np.single)
			placeholder = codes.copy()
		else:
			codes = codes.astype(np.double)
			placeholder = codes.copy()


		for j in range(self.p):
			for i in range(1, self.N):
				codes[i,j] = self._float_xor(placeholder[i-1,j],codes[i,j])


		codes = codes.astype(np.float32).flatten(order='F')

		fname = self.CODES
		np.save(fname, codes)

		compressz(self.CODES + '.npy', self.CODES+'.npyz')
		self.DATA_FILES += [self.CODES + '.npyz']


		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.CODES + '.npyz', self.CODES+'.npy')
		codes = np.load(self.CODES+".npy")

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		codes = codes.reshape(N,p, order='F').astype(np.float64)
		coderange = np.max(codes)

		
		for i in range(1, self.N):
			for j in range(p):
				codes[i,j] = self._float_xor(codes[i-1,j],codes[i,j])
				#print(codes[i,j])
		
		for j in range(p):
			codes[:,j] = (codes[:,j])*(normalization[0,j] - normalization[1,j]) + normalization[1,j]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


	#zero out as many bits as possible to hit error threshold
	def _strip_code(self, vector):
		p = vector.shape[0]

		for i in range(p): #go component by component
			value = vector[i]
			ba = bytearray(struct.pack("d", value))

			for j in range(len(ba)):
				tmp = ba[j]
				ba[j] = int('00000000')
				newvalue = struct.unpack("d", ba)[0]

				if np.abs(newvalue - value) > self.error_thresh:
					ba[j] = tmp
					vector[i] = struct.unpack("d", ba)[0]
					break

		return vector, j


		#zero out as many bits as possible to hit error threshold
	def _float_xor(self, v1, v2):
		ba1 = bytearray(struct.pack("d", v1))
		ba2 = bytearray(struct.pack("d", v2))


		for j in range(len(ba1)):
			ba2[j] = ba1[j] ^ ba2[j]
		return struct.unpack("d", ba2)[0]
