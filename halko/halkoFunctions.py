"""
Fast implementation of PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import ceil

# Import own scripts
from halko import shared_cy

### PCAone Halko with full data in memory ###
def halko(file, M, N, K, p_iter, threads):
	Bi = ceil(N/4) # Length of bytes to describe n individuals
	A = np.zeros((M, N))
	f = np.zeros(M)
	c = np.zeros(M)
	with open(file, "rb") as bed: # Read binary bed-file
		D = np.fromfile(bed, dtype=np.uint8, offset=3)
	shared_cy.plinkLoad(D, A, f, c, Bi, N, M, threads)
	del D, f, c
	Omg = np.random.standard_normal(size=(N, K+10))
	for p in range(p_iter):
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
		G = np.dot(A, Omg)
		H = np.dot(A.T, G)
	Q, R = np.linalg.qr(G, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, s, V = np.linalg.svd(B, full_matrices=False)
	del B
	U = np.dot(Q, Uhat)
	return U[:,:K], s[:K], V[:K,:]

### Out-of-core batched PCAone Halko ###
def halkoBatch(file, M, N, B, K, p_iter, threads):
	L = K + 10
	Bi = ceil(N/4) # Length of bytes to describe n individuals
	Omg = np.random.standard_normal(size=(N, L))
	G = np.zeros((M, L))
	H = np.zeros((N, L))
	for p in range(p_iter):
		A = np.zeros((B, N))
		f = np.zeros(B)
		c = np.zeros(B)
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(ceil(M/B)):
			if (b+1)*B >= M: # Last batch
				del A, f, c # Ensure no extra copy
				A = np.zeros((M - b*B, N))
				f = np.zeros(M - b*B)
				c = np.zeros(M - b*B)
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=(M-b*B)*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, N, M-b*B, threads)
				del D
				G[(b*B):M] = np.dot(A, Omg)
				H += np.dot(A.T, G[(b*B):M])
				del A, f, c # Ensure no extra copy
			else:
				with open(file, "rb") as bed: # Read binary bed-file
					D = np.fromfile(bed, dtype=np.uint8, count=B*Bi, offset=3 + (b*B)*Bi)
				shared_cy.plinkLoad(D, A, f, c, Bi, N, B, threads)
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
	return U[:,:K], s[:K], V[:K,:]
