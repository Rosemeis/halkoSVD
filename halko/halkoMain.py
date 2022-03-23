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
parser = argparse.ArgumentParser(prog="HalkoGPU")
parser.add_argument("-p", "--plink", metavar="FILE-PREFIX",
                    help="Prefix PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--n_eig", metavar="INT", type=int, default=10,
                    help="Number of eigenvectors to extract")
parser.add_argument("-b", "--batch", metavar="INT", type=int, default=8192,
                    help="Number of SNPs per batch - batch-size")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
                    help="Number of threads to use (CPU only)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="halko",
                    help="Prefix output name")
parser.add_argument("--power", metavar="INT", type=int, default=5,
                    help="Number of power iterations to perform")
parser.add_argument("--cpu", action="store_true",
                    help="Use batch-based CPU instead of GPU")
parser.add_argument("--full", action="store_true",
                    help="Full CPU based model. Overrides --cpu, --batch")
parser.add_argument("--loadings", action="store_true",
                    help="Save loadings")

##### PCAone Halko #####
def main():
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()
    print("PCAone Halko CPU/GPU implementation")
    assert args.plink is not None, "No input data (-plink)!"
    if args.cpu or args.full:
        if args.full:
            print("Full CPU implementation")
        else:
            print("Batch-based CPU implementation using a batch-size of {}".format(args.batch))
    else:
        print("Batch-based GPU implmentation using a batch-size of {}".format(args.batch))

    # Control threads
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Import numerical libraries
    import numpy as np
    import cupy as cp

    # Import own scripts
    from halko import halkoFunctions

    # Finding length of .fam and .bim files
    n = extract_length(args.plink + ".fam")
    m = extract_length(args.plink + ".bim")
    print("Data: {} samples, {} SNPs".format(n, m))

    # Perform SVD
    print("Extracting {} eigenvectors".format(args.n_eig))
    if args.cpu or args.full:
        if args.full:
            U, s, V = halkoFunctions.halkoCPU(args.plink + ".bed", m, n, \
						args.n_eig, args.power, args.threads)
        else:
            U, s, V = halkoFunctions.halkoCPUbatch(args.plink + ".bed", m, n, \
						args.batch, args.n_eig, args.power, args.threads)
    else:
        U, s, V = halkoFunctions.halkoGPUbatch(args.plink + ".bed", m, n, \
						args.batch, args.n_eig, args.power, args.threads)

    # Save matrices
    np.savetxt(args.out + ".eigenvecs", V.T, fmt="%.7f")
    print("Saved eigenvector(s) as {}.eigenvecs".format(args.out))
    np.savetxt(args.out + ".eigenvals", s**2/float(m), fmt="%.7f")
    print("Saved eigenvalue(s) as {}.eigenvals".format(args.out))

    # Save loadings
    if args.loadings:
        np.savetxt(args.out + ".loadings", U, fmt="%.7f")
        print("Saved SNP loadings as {}.loadings".format(args.out))

##### Define main #####
if __name__ == "__main__":
    main()
