"""
Main caller for CPU and GPU implementations of PCAone Halko algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
import subprocess

# Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

# Argparse
parser = argparse.ArgumentParser(prog="halko")
parser.add_argument("-b", "--bfile", metavar="FILE-PREFIX",
	help="Prefix PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--n_eig", metavar="INT", type=int, default=10,
	help="Number of eigenvectors to extract")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads to use")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="halko",
	help="Prefix output name")
parser.add_argument("--batch", metavar="INT", type=int,
	help="Perform out-of-core Halko SVD with provided batch-size")
parser.add_argument("--power", metavar="INT", type=int, default=7,
	help="Number of power iterations to perform")
parser.add_argument("--loadings", action="store_true",
	help="Save loadings")

##### PCAone Halko #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("Fast PCAone Halko implementation")
	assert args.bfile is not None, "No input data (--bfile)!"
	if args.batch is not None:
		print(f"Using out-of-core algorithm with a batch-size of {args.batch}")

	# Control threads
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries
	import numpy as np
	from halko import halkoFunctions

	# Finding length of .fam and .bim files
	N = extract_length(f"{args.bfile}.fam")
	M = extract_length(f"{args.bfile}.bim")
	print(f"Data: {N} samples, {M} SNPs")

	# Perform SVD
	print(f"Extracting {args.n_eig} eigenvectors")
	if args.batch is None:
		U, S, V = halkoFunctions.halko(args.bfile + ".bed", M, N, \
			args.n_eig, args.power, args.threads)
	else:
		U, S, V = halkoFunctions.halkoBatch(args.bfile + ".bed", M, N, \
			args.batch, args.n_eig, args.power, args.threads)

	# Save matrices
	np.savetxt(args.out + ".eigenvecs", V.T, fmt="%.7f")
	print(f"Saved eigenvector(s) as {args.out}.eigenvecs")
	np.savetxt(args.out + ".eigenvals", S**2/float(M), fmt="%.7f")
	print(f"Saved eigenvalue(s) as {args.out}.eigenvals")

	# Save loadings
	if args.loadings:
		np.savetxt(args.out + ".loadings", U, fmt="%.7f")
		print("Saved SNP loadings as {}.loadings".format(args.out))



##### Define main #####
if __name__ == "__main__":
	main()
