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

# Batched randomized SVD with dynamic shifts
def randomizedSVD(G, f, d, K, batch, power, rng):
	M, N = G.shape
	W = ceil(M/batch)
	a = 0.0
	L = K + 10
	A = np.zeros((M, L))
	H = np.zeros((N, L))
	X = np.zeros((batch, N))
	O = rng.standard_normal(size=(M, L))

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared.plinkChunk(G, X, f, d, M_w)
		H += np.dot(X.T, O[M_w:(M_w + X.shape[0])])
	Q, _ = np.linalg.qr(H)
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
		Q, S, _ = np.linalg.svd(H - a*Q, full_matrices=False)
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
	U, S, V = np.linalg.svd(A, full_matrices=False)
	print(".\n")
	return U[:,:K], S[:K], np.dot(Q, V)[:,:K]
