# Cython/Python implementation of Halko algorithm (PCAone)
This is a fast implementation of the PCAone Halko algorithm in Python/Cython for genotype data. It takes binary PLINK format (*.bed, *.bim, *.fam) as input. For simplicity, mean imputation is performed for missing data.

It is inspired by the lovely *PCAone* software! Have a look [here](https://github.com/Zilong-Li/PCAone).

## Installation
```bash
# Option 1: Build and install via PyPI
pip install halkoSVD

# Option 2: Download source and install via pip
git clone https://github.com/Rosemeis/halkoSVD.git
cd halkoSVD
pip install .

# Option 3: Download source and install in a new Conda environment
git clone https://github.com/Rosemeis/halkoSVD.git
conda env create -f halkoSVD/environment.yml
conda activate halkoSVD
```
You can now run the program with the `halkoSVD` command.

## Quick usage
Provide `halkoSVD` with the file prefix of the PLINK files.
```bash
# Check help message of the program
halkoSVD -h

# Extract the top 10 PCs
halkoSVD --bfile input --threads 32 --pca 10 --out halko
```

### Options
* `--pcaone`, perform fast PCAone block iterations
* `--seed`, set random seed for reproducibility (42)
* `--power`, specify the number of power iterations (11)
* `--batch`, specify the batch size to process SNPs (8192)
* `--loadings`, save the SNP loadings
* `--raw`, only output eigenvectors without FID/IID
