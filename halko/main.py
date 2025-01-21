"""
halkoSVD.
Main caller for Python/Cython Halko randomized SVD implementation.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from time import time

# Argparse
parser = argparse.ArgumentParser(prog="halkoSVD")
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
parser.add_argument("--power", metavar="INT", type=int, default=10,
	help="Number of power iterations to perform (10)")
parser.add_argument("--batch", metavar="INT", type=int, default=8192,
	help="Mini-batch size for randomized SVD (8192)")
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
	print(f"Fast randomized SVD implementation using {args.threads} thread(s).")
	print(f"Computing randomized SVD with a batch-size of {args.batch} SNPs.")

	# Check input
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.pca > 0, "Please select a valid number of PCs > 0!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.seed >= 0, "Please select a valid seed!"
	assert args.power > 0, "Please select a valid number of power iterations!"
	assert args.batch > 0, "Please select a valid value for batch size!"
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

	# Reading PLINK files
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	G, f, d = functions.readPlink(args.bfile)
	print(f"\rLoaded {G.shape[1]} samples and {G.shape[0]} SNPs.\n")

	# Perform Randomized SVD
	print(f"Extracting {args.pca} eigenvectors.")
	rng = np.random.default_rng(args.seed)
	U, S, V = functions.randomizedSVD(G, f, d, args.pca, args.batch, args.power, rng)
	del G, f, d

	# Save matrices
	if args.raw:
		np.savetxt(f"{args.out}.eigenvecs", V, fmt="%.7f")
	else:
		F = np.loadtxt(f"{args.bfile}.fam", usecols=[0,1], dtype=np.str_)
		h = ["#FID", "IID"] + [f"PC{k}" for k in range(1, args.pca+1)]
		V = np.hstack((F, np.round(V, 7)))
		np.savetxt(f"{args.out}.eigenvecs", V, fmt="%s", delimiter="\t", \
			header="\t".join(h), comments="")
	print(f"Saved eigenvector(s) as {args.out}.eigenvecs")
	np.savetxt(f"{args.out}.eigenvals", S**2/float(U.shape[0]), fmt="%.7f")
	print(f"Saved eigenvalue(s) as {args.out}.eigenvals")

	# Save loadings
	if args.loadings:
		np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
		print(f"Saved SNP loadings as {args.out}.loadings")
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time {t_min}m{t_sec}s")
