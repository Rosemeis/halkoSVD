"""
Main caller for PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from math import ceil
from time import time

# Argparse
parser = argparse.ArgumentParser(prog="halko")
parser.add_argument("-b", "--bfile", metavar="PLINK",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--pca", metavar="INT", type=int, default=10,
	help="Number of eigenvectors to extract (10)")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("-s", "--seed", metavar="INT", type=int, default=42,
	help="Set random seed (42)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="halko",
	help="Prefix output name (halko)")
parser.add_argument("--power", metavar="INT", type=int, default=11,
	help="Number of power iterations to perform (11)")
parser.add_argument("--batch", metavar="INT", type=int, default=4096,
	help="Mini-batch size for randomized SVD (4096)")
parser.add_argument("--full", action="store_true",
	help="Perform full randomized SVD in memory")
parser.add_argument("--loadings", action="store_true",
	help="Save loadings")
parser.add_argument("--raw", action="store_true",
	help="Raw output without '*.fam' info")



##### PCAone Halko #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print(f"Fast PCAone Halko implementation using {args.threads} thread(s).")
	if args.full:
		print("Computing randomized SVD with data in memory.")
	else:
		print(f"Computing randomized SVD with a batch-size of {args.batch} SNPs.")
	assert args.bfile is not None, "No input data (--bfile)!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from halko import functions
	from halko import shared

	# Finding length of .fam and .bim files
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	G, M, N = functions.readPlink(args.bfile)
	print(f"\rLoaded {N} samples and {M} SNPs.")

	# Estimate allele frequencies and scaling
	f = np.zeros(M)
	L = np.zeros((M, 3))
	shared.estimateFreq(G, f, N, args.threads)
	assert (np.min(f) > 0.0) & (np.max(f) < 1.0), "Please perform MAF filtering!"
	shared.createLookup(L, f, args.threads)
	del f

	# Perform Randomized SVD
	print(f"Extracting {args.pca} eigenvectors.")
	if args.full:
		U, S, V = functions.fullSVD(G, L, N, args.pca, args.power, args.seed, \
			args.threads)
	else:
		U, S, V = functions.batchSVD(G, L, N, args.pca, args.batch, args.power, \
			args.seed, args.threads)

	# Save matrices
	if args.raw:
		np.savetxt(f"{args.out}.eigenvecs", V.T, fmt="%.7f")
	else:
		F = np.loadtxt(f"{args.bfile}.fam", usecols=[0,1], dtype=np.str_)
		V = np.hstack((F, np.round(V.T, 7)))
		np.savetxt(f"{args.out}.eigenvecs", V, fmt="%s")
	print(f"Saved eigenvector(s) as {args.out}.eigenvecs")
	np.savetxt(f"{args.out}.eigenvals", S**2/float(M), fmt="%.7f")
	print(f"Saved eigenvalue(s) as {args.out}.eigenvals")

	# Save loadings
	if args.loadings:
		np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
		print(f"Saved SNP loadings as {args.out}.loadings")
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time {t_min}m{t_sec}s")
