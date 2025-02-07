# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

# Estimate minor allele frequencies
cpdef void expandGeno(const unsigned char[:,::1] D, unsigned char[:,::1] G, \
		double[::1] f, double[::1] d) noexcept nogil:
	cdef:
		size_t M = D.shape[0]
		size_t B = D.shape[1]
		size_t N = G.shape[1]
		size_t b, g, i, j, bytepart
		double n
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M):
		i = 0
		n = 0.0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				G[j,i] = recode[byte & mask]
				if G[j,i] != 9:
					f[j] += <double>G[j,i]
					n = n + 1.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f[j] /= <double>(2.0*n)
		d[j] = 1.0/sqrt(2.0*f[j]*(1.0 - f[j]))

# Load standardized chunk from PLINK file for Halko randomixed SVD
cpdef void plinkChunk(const unsigned char[:,::1] G, double[:,::1] X, double[::1] f, \
		double[::1] d, const size_t M_w) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t i, j, k
		double fk, dk
	for j in prange(M):
		k = M_w + j
		fk = f[k]
		dk = d[k]
		for i in range(N):
			if G[k,i] != 9:
				X[j,i] = (<double>G[k,i] - 2.0*fk)*dk
			else:
				X[j,i] = 0.0
