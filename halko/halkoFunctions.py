"""
CPU and GPU implementations of PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import cupy as cp
from math import ceil

# Import own scripts
from halko import shared_cy

### PCAone Halko with full data in memory ###
def halkoCPU(file, m, n, k, n_iter, threads):
	Bi = ceil(n/4) # Length of bytes to describe n individuals
	A = np.zeros((m, n), dtype=np.float32)
	f = np.zeros(m, dtype=np.float32)
	c = np.zeros(m, dtype=np.int32)
	with open(file, "rb") as bed: # Read binary bed-file
		D = np.fromfile(bed, dtype=np.uint8, offset=3)
	shared_cy.plinkLoad(D, A, f, c, Bi, n, m, threads)
	del D, f, c
	Omg = np.random.standard_normal(size=(n, k+10)).astype(np.float32)
	for p in range(n_iter):
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
		G = np.dot(A, Omg)
		H = np.dot(A.T, G)
	Q, R = np.linalg.qr(G, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, s, V = np.linalg.svd(B, full_matrices=False)
	del B
	U = np.dot(Q, Uhat)
	return U[:,:k], s[:k], V[:k,:]

### Out-of-core batched PCAone Halko for CPU ###
def halkoCPUbatch(file, m, n, B, k, n_iter, threads):
	l = k + 10
	Bi = ceil(n/4) # Length of bytes to describe n individuals
	Omg = np.random.standard_normal(size=(n, l)).astype(np.float32)
	G = np.zeros((m, l), dtype=np.float32)
	H = np.zeros((n, l), dtype=np.float32)
	for p in range(n_iter):
		A = np.zeros((B, n), dtype=np.float32)
		f = np.zeros(B, dtype=np.float32)
		c = np.zeros(B, dtype=np.int32)
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(ceil(m/B)):
			if (b+1)*B >= m: # Last batch
				del A, f, c # Ensure no extra copy
				A = np.zeros((m - b*B, n), dtype=np.float32)
				f = np.zeros(m - b*B, dtype=np.float32)
				c = np.zeros(m - b*B, dtype=np.int32)
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=(m-b*B)*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, n, m-b*B, threads)
				del D
				G[(b*B):m] = np.dot(A, Omg)
				H += np.dot(A.T, G[(b*B):m])
				del A, f, c # Ensure no extra copy
			else:
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=B*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, n, B, threads)
				del D
				G[(b*B):((b+1)*B)] = np.dot(A, Omg)
				H += np.dot(A.T, G[(b*B):((b+1)*B)])
				f.fill(0.0)
				c.fill(0)
	Q, R = np.linalg.qr(G, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, s, V = np.linalg.svd(B, full_matrices=False)
	del B
	U = np.dot(Q, Uhat)
	return U[:,:k], s[:k], V[:k,:]

### Out-of-core batched PCAone Halko for GPU ###
def halkoGPUbatch(file, m, n, B, k, n_iter, threads):
	l = k + 10
	Bi = ceil(n/4) # Length of bytes to describe n individuals
	Omg = cp.random.standard_normal(size=(n, l), dtype=cp.float32)
	G = cp.zeros((m, l), dtype=cp.float32)
	H = cp.zeros((n, l), dtype=cp.float32)
	for p in range(n_iter):
		A = np.zeros((B, n), dtype=np.float32)
		f = np.zeros(B, dtype=np.float32)
		c = np.zeros(B, dtype=np.int32)
		if p > 0:
			Omg, _ = cp.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(ceil(m/B)):
			if (b+1)*B >= m:
				del A, f, c # Ensure no extra copy
				A = np.zeros((m - b*B, n), dtype=np.float32)
				f = np.zeros(m - b*B, dtype=np.float32)
				c = np.zeros(m - b*B, dtype=np.int32)
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=(m-b*B)*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, n, m-b*B, threads)
				del D
				A_gpu = cp.asarray(A)
				G[(b*B):m] = cp.dot(A_gpu, Omg)
				H += cp.dot(A_gpu.T, G[(b*B):m])
				del A, f, c, A_gpu # Ensure no extra copy
			else:
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=B*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, n, B, threads)
				del D
				A_gpu = cp.asarray(A)
				G[(b*B):((b+1)*B)] = cp.dot(A_gpu, Omg)
				H += cp.dot(A_gpu.T, G[(b*B):((b+1)*B)])
				f.fill(0.0)
				c.fill(0)
				del A_gpu
	Q, R = cp.linalg.qr(G, mode="reduced")
	B = cp.linalg.solve(R.T, H.T)
	Uhat, s, V = cp.linalg.svd(B, full_matrices=False)
	del B
	U = cp.dot(Q, Uhat)
	return cp.asnumpy(U[:,:k]), cp.asnumpy(s[:k]), cp.asnumpy(V[:k,:])
