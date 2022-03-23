# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, fabs

# Load and generate standardized batch array
cpdef plinkLoad(unsigned char[:,::1] D, float[:,::1] A, float[::1] f,
				int[::1] c, int Bi, int n, int m, int t):
	cdef signed char[4] recode = [2, 9, 1, 0]
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						c[j] = c[j] + 1
						f[j] = f[j] + <float>code
					A[j, i] = <float>code
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n: # Estimate allele frequency and move to next SNP
						if c[j] > 0:
							f[j] = f[j]/(2.0*<float>c[j])
						else:
							f[j] = 0.0
						break
			# Standardize the loaded batch
			for k in range(n):
				if fabs(A[j, k] - 9.0) < 1e-4:
					A[j, k] = 0.0 # Mean imputation
				else:
					A[j, k] = (A[j, k] - 2.0*f[j])/sqrt(2.0*f[j]*(1.0 - f[j]))
