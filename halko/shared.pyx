# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

# Estimate minor allele frequencies
cpdef void estimateFreq(unsigned char[:,::1] G, double[::1] f, int N, int t) \
		noexcept nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int i, j, b, bytepart
		double n
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		n = 0.0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					f[j] += <double>recode[byte & mask]
					n = n + 1.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f[j] /= (2.0*n)

# Load standardized chunk from PLINK file for SVD
cpdef void plinkChunk(unsigned char[:,::1] G, double[:,::1] X, double[::1] f, \
		double[::1] s, int M_b, int t) noexcept nogil:
	cdef:
		int M = X.shape[0]
		int N = X.shape[1]
		int B = G.shape[1]
		int d, i, j, b, bytepart
		float a, m
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		d = M_b + j
		a = s[d]
		m = 2.0*f[d]
		i = 0		
		for b in range(B):
			byte = G[d,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					X[j,i] = (<double>recode[byte & mask] - m)*a
				else:
					X[j,i] = 0.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
