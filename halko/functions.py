"""
Fast implementation of PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import ceil

# Import own scripts
from halko import shared

### Read PLINK files
def readPlink(bfile):
	# Find length of fam-file
	N = 0
	with open(f"{bfile}.fam", "r") as fam:
		for _ in fam:
			N += 1
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals

	# Read .bed file
	with open(f"{bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (G.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = G.shape[0]//N_bytes
	G.shape = (M, N_bytes)
	return G, M, N

### Mini-batch randomized SVD (PCAone Halko)
def batchSVD(G, L, N, K, batch, power, seed, threads):
	M = G.shape[0]
	W = ceil(M/batch)
	D = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, D))
	A = np.zeros((M, D))
	H = np.zeros((N, D))
	for p in range(power):
		X = np.zeros((batch, N))
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in range(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				del X # Ensure no extra copy
				X = np.zeros((M - M_w, N))
			shared.plinkChunk(G, L, X, M_w, threads)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, X
	return U[:,:K], S[:K], V[:K,:]


### Full randomized SVD (PCAone Halko)
def fullSVD(G, L, N, K, power, seed, threads):
	M = G.shape[0]
	D = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, D))
	A = np.zeros((M, D))
	H = np.zeros((N, D))
	X = np.zeros((M, N))
	shared.plinkChunk(G, L, X, 0, threads)
	for p in range(power):
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")			
		np.dot(X, O, out=A)
		np.dot(X.T, A, out=H)
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, X
	return U[:,:K], S[:K], V[:K,:]
