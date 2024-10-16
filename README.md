# Python implementation of Halko algorithm (PCAone)
This is a fast implementation of the PCAone (H+Y) Halko algorithm in Python/Cython for genetic data. It takes binary PLINK format (*.bed, *.bim, *.fam) as input. For simplicity, mean imputation is performed for missing data.

It is inspired by the lovely *PCAone* software! Have a look [here](https://github.com/Zilong-Li/PCAone).

## Installation
```bash
# Vuild and install via PyPI
pip install halkoSVD

# Download source and install via pip
git clone https://github.com/Rosemeis/halkoSVD.git
cd halkoSVD
pip install .

# Download source and install in new Conda environment
git clone https://github.com/Rosemeis/halkoSVD.git
conda env create -f environment.yml
conda activate halkoSVD


# You can now run the program with the `halkoSVD` command
```

## Quick usage
Provide `halkoSVD` with the file prefix of the PLINK files.
```bash
# Check help message of the program
halkoSVD -h

# Extract the top 10 PCs
halkoSVD --bfile input --threads 32 --pca 10 --out halko
```

### Options
* `--power`, specify the number of power iterations (12)
* `--extra`, number of extra vectors for oversampling (16)
* `--batch`, specify the batch size to process SNPs (4096)
* `--full`, load the entire genotype matrix into matrix
* `--loadings`, save the SNP loadings
* `--raw`, only output eigenvectors without FID/IID
