# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

# Estimate minor allele frequencies
cpdef void createLookup(const unsigned char[:,::1] G, double[:,::1] L, const size_t N) \
		noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t B = G.shape[1]
		size_t b, g, i, j, bytepart
		double d, f, n
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M):
		# Estimate allele frequency
		i = 0
		f = 0.0
		n = 0.0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					f += <double>recode[byte & mask]
					n = n + 1.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f = f/<double>(2.0*n)
		d = 1.0/sqrt(2.0*f*(1.0 - f))

		# Fill look-up table
		L[j,0] = (0.0 - 2.0*f)*d
		L[j,1] = (1.0 - 2.0*f)*d
		L[j,2] = (2.0 - 2.0*f)*d

# Load standardized chunk from PLINK file for Halko randomixed SVD
cpdef void plinkChunk(const unsigned char[:,::1] G, const double[:,::1] L, \
		double[:,::1] X, const size_t M_b) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t b, d, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
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

# Load standardized chunk from PLINK file for PCAone randomixed SVD
cpdef void plinkBlock(const unsigned char[:,::1] G, const double[:,::1] L, \
		double[:,::1] X, const unsigned int[::1] s, const size_t M_b) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t b, d, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		d = s[M_b + j]
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