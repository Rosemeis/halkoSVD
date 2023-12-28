"""
Fast implementation of PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import subprocess
import numpy as np
from math import ceil

# Import own scripts
from halko import shared

### Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])


### Mini-batch randomized SVD (PCAone Halko)
def batchSVD(G, f, N, K, batch, power, seed, threads):
	M = G.shape[0]
	W = ceil(M/batch)
	L = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L))
	A = np.zeros((M, L))
	H = np.zeros((N, L))
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
			shared.plinkChunk(G, X, f, M_w, threads)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R = np.linalg.qr(A, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, X
	return U[:,:K], S[:K], V[:K,:]


### Full randomized SVD (PCAone Halko)
def fullSVD(G, f, N, K, power, seed, threads):
	M = G.shape[0]
	L = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L))
	A = np.zeros((M, L))
	H = np.zeros((N, L))
	X = np.zeros((M, N))
	shared.plinkChunk(G, X, f, 0, threads)
	for p in range(power):
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")			
		np.dot(X, O, out=A)
		np.dot(X.T, A, out=H)
	Q, R = np.linalg.qr(A, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, X
	return U[:,:K], S[:K], V[:K,:]