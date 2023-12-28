# Python implementation of Halko algorithm (PCAone)
This is an implementation of the PCAone Halko algorithm in Python/Cython. It takes binary PLINK format (*.bed, *.bim, *.fam) as input. For simplicity, mean imputation is performed for missing data.

It is inspired by the lovely *PCAone* software! Have a look [here](https://github.com/Zilong-Li/PCAone).

## Install and build
```bash
# Download and install
git clone https://github.com/Rosemeis/halkoSVD.git
cd halkoSVD
pip3 install .

# You can now run the program with the `halko` command
```

## Quick usage
Provide `halko` with the file prefix of the PLINK files.
```bash
# Check help message of the program
halko -h

# Extract top 10 PCs with a mini-batch size of 8192 SNPs
halko --bfile input --threads 32 --pca 10 --batch 8192 --out halko
```
