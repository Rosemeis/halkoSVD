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

### Function to load standardized batch of PLINK bed-file
def plinkBatch(file, start, end, n, threads):
	Bi = ceil(n/4) # Length of bytes to describe n individuals

	# Initiate genotype matrix
	A = np.zeros((end-start, n), dtype=np.float32)
	f = np.zeros(end-start, dtype=np.float32)
	c = np.zeros(end-start, dtype=np.int32)

	# Read binary bed-file
	with open(file, "rb") as bed:
		D = np.fromfile(bed, dtype=np.uint8, count=(end-start)*Bi, offset=3 + start*Bi)
	D = D.reshape((end-start, Bi))
	shared_cy.plinkLoad(D, A, f, c, Bi, n, end-start, threads)
	return A

### PCAone Halko with full data in memory ###
def halkoCPU(file, m, n, k, n_iter, threads):
	A = plinkBatch(file, 0, m, n, threads)
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
	Omg = np.random.standard_normal(size=(n, l)).astype(np.float32)
	G = np.zeros((m, l), dtype=np.float32)
	H = np.zeros((n, l), dtype=np.float32)
	for p in range(n_iter):
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(ceil(m/B)):
			if (b+1)*B >= m:
				A_cpu = plinkBatch(file, b*B, m, n, threads)
				G[(b*B):m] = np.dot(A_cpu, Omg)
				H += np.dot(A_cpu.T, G[(b*B):m])
			else:
				A_cpu = plinkBatch(file, b*B, (b+1)*B, n, threads)
				G[(b*B):((b+1)*B)] = np.dot(A_cpu, Omg)
				H += np.dot(A_cpu.T, G[(b*B):((b+1)*B)])
			del A_cpu
	Q, R = np.linalg.qr(G, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, s, V = np.linalg.svd(B, full_matrices=False)
	del B
	U = np.dot(Q, Uhat)
	return U[:,:k], s[:k], V[:k,:]

### Out-of-core batched PCAone Halko for GPU ###
def halkoGPUbatch(file, m, n, B, k, n_iter, threads):
	l = k + 10
	Omg = cp.random.standard_normal(size=(n, l), dtype=cp.float32)
	G = cp.zeros((m, l), dtype=cp.float32)
	H = cp.zeros((n, l), dtype=cp.float32)
	for p in range(n_iter):
		if p > 0:
			Omg, _ = cp.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(ceil(m/B)):
			if (b+1)*B >= m:
				A_gpu = cp.asarray(plinkBatch(file, b*B, m, n, threads))
				G[(b*B):m] = cp.dot(A_gpu, Omg)
				H += cp.dot(A_gpu.T, G[(b*B):m])
			else:
				A_gpu = cp.asarray(plinkBatch(file, b*B, (b+1)*B, n, threads))
				G[(b*B):((b+1)*B)] = cp.dot(A_gpu, Omg)
				H += cp.dot(A_gpu.T, G[(b*B):((b+1)*B)])
			del A_gpu
	Q, R = cp.linalg.qr(G, mode="reduced")
	B = cp.linalg.solve(R.T, H.T)
	Uhat, s, V = cp.linalg.svd(B, full_matrices=False)
	del B
	U = cp.dot(Q, Uhat)
	return cp.asnumpy(U[:,:k]), cp.asnumpy(s[:k]), cp.asnumpy(V[:k,:])
