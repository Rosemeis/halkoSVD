# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

# Estimate minor allele frequencies
cpdef void estimateFreq(const unsigned char[:,::1] G, double[::1] f, const int N, \
		const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int b, i, j, bytepart
		double n
		unsigned char[4] recode = [2, 9, 1, 0]
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

# Create look-up table of standardized genotypes
cpdef void createLookup(double[:,::1] L, const double[::1] f, const int t) \
		noexcept nogil:
	cdef:
		int M = L.shape[0]
		int G = L.shape[1]
		int g, j
	for j in prange(M, num_threads=t):
		for g in range(3):
			L[j,g] = (<double>g - 2.0*f[j])/sqrt(2.0*f[j]*(1.0 - f[j]))

# Load standardized chunk from PLINK file for SVD
cpdef void plinkChunk(const unsigned char[:,::1] G, const double[:,::1] L, \
		double[:,::1] X, const int M_b, const int t) noexcept nogil:
	cdef:
		int M = X.shape[0]
		int N = X.shape[1]
		int B = G.shape[1]
		int b, d, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M, num_threads=t):
		d = M_b + j
		i = 0		
		for b in range(B):
			byte = G[d,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = L[d,g]
				else:
					X[j,i] = 0.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
