# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdint cimport uint8_t, uint32_t

# Estimate minor allele frequencies
cpdef void expandGeno(
		const uint8_t[:,::1] D, uint8_t[:,::1] G, double[::1] f, double[::1] d
	) noexcept nogil:
	cdef:
		uint8_t[4] recode = [2, 9, 1, 0]
		uint8_t mask = 3
		uint8_t byte
		uint8_t* g
		uint32_t M = D.shape[0]
		uint32_t B = D.shape[1]
		uint32_t N = G.shape[1]
		double n
		size_t b, i, j, bytepart
	for j in prange(M):
		i = 0
		n = 0.0
		g = &G[j,0]
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				g[i] = recode[byte & mask]
				if g[i] != 9:
					f[j] += <double>g[i]
					n = n + 1.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f[j] /= <double>(2.0*n)
		d[j] = 1.0/sqrt(2.0*f[j]*(1.0 - f[j]))

# Load standardized chunk from PLINK file for Halko randomixed SVD
cpdef void plinkChunk(
		uint8_t[:,::1] G, double[:,::1] X, const double[::1] f, const double[::1] d, const uint32_t M_w
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = X.shape[0]
		uint32_t N = X.shape[1]
		double fk, dk
		double* x
		size_t i, j, k
	for j in prange(M):
		k = M_w + j
		fk = f[k]
		dk = d[k]
		g = &G[k,0]
		x = &X[j,0]
		for i in range(N):
			if g[i] != 9:
				x[i] = (<double>g[i] - 2.0*fk)*dk
			else:
				x[i] = 0.0
