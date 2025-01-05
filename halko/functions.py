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
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (G.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = G.shape[0]//N_bytes
	G.shape = (M, N_bytes)
	return G, M, N

# Mini-batch randomized SVD
def halkoSVD(G, L, N, K, batch, power, rng):
	M = G.shape[0]
	W = ceil(M/batch)
	D = K + 10
	A = np.zeros((M, D))
	H = np.zeros((N, D))
	X = np.zeros((batch, N))
	O = rng.standard_normal(size=(N, D))
	for p in np.arange(power):
		print(f"\rPower iteration {p+1}/{power}", end="")
		X = np.zeros((batch, N))
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in np.arange(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				X = np.zeros((M - M_w, N))
			shared.plinkChunk(G, L, X, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	C = np.linalg.solve(R.T, H.T)
	U_hat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, U_hat)
	print(".\n")
	return U[:,:K], S[:K], V[:K,:]

# PCAone randomized SVD
def pcaoneSVD(G, L, N, K, batch, rng):
	M = G.shape[0]
	D = K + 10
	B = 64
	H = np.zeros((N, D))
	S = np.arange(M, dtype=np.uint32)
	O = rng.standard_normal(size=(N, D))

	# PCAone block power iterations
	for e in np.arange(6):
		print(f"\rEpoch {e+1}/7", end="")
		rng.shuffle(S)
		A = np.zeros((ceil(M/B), D))
		for b in np.arange(B):
			s = S[(b*A.shape[0]):min((b+1)*A.shape[0], M)]
			W = ceil(s.shape[0]/batch)
			X = np.zeros((batch, N))
			if b == (B-1):
				A = np.zeros((s.shape[0], D))
			if ((e == 0) and (b > 0)) or (e > 0):
				O, _ = np.linalg.qr(H, mode="reduced")
				H.fill(0.0)
			for w in np.arange(W):
				M_w = w*batch
				if w == (W-1): # Last batch
					X = np.zeros((s.shape[0] - M_w, N))
				shared.plinkBlock(G, L, X, s, M_w)
				A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
				H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		B = B//2
	
	# Standard power iteration
	print("\rEpoch 7/7", end="")
	W = ceil(M/batch)
	A = np.zeros((M, D))
	X = np.zeros((batch, N))
	O, _ = np.linalg.qr(H, mode="reduced")
	H.fill(0.0)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared.plinkChunk(G, L, X, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	C = np.linalg.solve(R.T, H.T)
	U_hat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, U_hat)
	print(".\n")
	return U[:,:K], S[:K], V[:K,:]