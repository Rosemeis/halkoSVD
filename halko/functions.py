import numpy as np
from math import ceil
from halko import shared

# Read PLINK files
def readPlink(bfile):
	# Find length of fam-file
	N = 0
	with open(f"{bfile}.fam", "r") as fam:
		for _ in fam:
			N += 1
	N_bytes = ceil(N/4)

	# Read .bed file
	with open(f"{bfile}.bed", "rb") as bed:
		D = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (D.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = D.shape[0]//N_bytes
	D.shape = (M, N_bytes)

	# Expand and estimate allele frequencies
	G = np.zeros((M, N), dtype=np.uint8)
	f = np.zeros(M)
	d = np.zeros(M)
	shared.expandGeno(D, G, f, d)
	return G, f, d

# SVD through eigendecomposition
def eigSVD(H):
	D, V = np.linalg.eigh(np.dot(H.T, H))
	S = np.sqrt(D)
	U = np.dot(H, V*(1.0/S))
	return np.ascontiguousarray(U[:,::-1]), np.ascontiguousarray(S[::-1]), np.ascontiguousarray(V[:,::-1])

# Batched randomized SVD with dynamic shift
def randomizedSVD(G, f, d, K, batch, power, rng):
	M, N = G.shape
	W = ceil(M/batch)
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L))
	X = np.zeros((batch, N))
	A = rng.standard_normal(size=(M, L))

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared.plinkChunk(G, X, f, d, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for p in np.arange(power):
		print(f"\rPower iteration {p+1}/{power}", end="")
		X = np.zeros((batch, N))
		for w in np.arange(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				X = np.zeros((M - M_w, N))
			shared.plinkChunk(G, X, f, d, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N))
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared.plinkChunk(G, X, f, d, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	print(".\n")
	return U[:,:K], S[:K], np.dot(Q, V)[:,:K]
