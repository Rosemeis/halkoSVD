# Python implementation of Halko algorithm (PCAone)
This is an implementation of the PCAone Halko algorithm in Python/Cython. It takes binary PLINK format (*.bed, *.bim, *.fam) as input. For simplicity, mean imputation is performed for missing data.

It is inspired by the lovely *PCAone* software! Have a look [here](https://github.com/Zilong-Li/PCAone).

## Install and build
```bash
# Install via PyPI
pip3 install halkoSVD

# Install via Conda
conda env create -f environment.yml

# Download and install from GitHub directly
git clone https://github.com/Rosemeis/halkoSVD.git
cd halkoSVD
pip3 install .

# You can now run the program with the `halkoSVD` command
```

## Quick usage
Provide `halkoSVD` with the file prefix of the PLINK files.
```bash
# Check help message of the program
halkoSVD -h

# Extract top 10 PCs with a mini-batch size of 8192 SNPs
halkoSVD --bfile input --threads 32 --pca 10 --batch 8192 --out halko
```
