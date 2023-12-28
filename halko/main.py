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
parser.add_argument("--batch", metavar="INT", type=int, default=10000,
	help="Mini-batch size for randomized SVD (10000)")
parser.add_argument("--loadings", action="store_true",
	help="Save loadings")



##### PCAone Halko #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print(f"Fast PCAone Halko implementation using {args.threads} thread(s).")
	print(f"Computing randomized SVD with a batch-size of {args.batch} SNPs.")
	assert args.bfile is not None, "No input data (--bfile)!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from halko import functions
	from halko import shared

	# Finding length of .fam and .bim files
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	N = functions.extract_length(f"{args.bfile}.fam")
	M = functions.extract_length(f"{args.bfile}.bim")
	with open(f"{args.bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	B = ceil(N/4)
	G.shape = (M, B)
	print(f"Loaded data: {N} samples, {M} SNPs")

	# Estimate allele frequencies
	f = np.zeros(M)
	shared.estimateFreq(G, f, N, args.threads)
	assert (np.min(f) > 0.0) & (np.max(f) < 1.0), "Please perform MAF filtering!"

	# Perform Randomized SVD
	print(f"Extracting {args.pca} eigenvectors.")
	U, S, V = functions.svd(G, f, N, args.batch, args.pca, args.power, \
		args.seed, args.threads)

	# Save matrices
	np.savetxt(f"{args.out}.eigenvecs", V.T, fmt="%.7f")
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
