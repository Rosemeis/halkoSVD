# CPU/GPU Python implementation of Halko algorithm (PCAone)
This is an implementation of the PCAone Halko algorithm, the fastest SVD for genetic data. The implementation is using the *cupy* python library as backend for GPU linear algebra computations (out-of-core). There is also a CPU based implementation in *NumPy* either batch-based (out-of-core) or loading full data into memory. This may be the fastest randomized SVD implementation in the world for genetic data.

It simply only uses PLINK files (*.bed, *.bim, *.fam) as input. For simplicity, I perform mean imputation for missing data.

It is inspired by the lovely *PCAone* software! Have a look [here](https://github.com/Zilong-Li/PCAone).

xoxo Jonas

## Download, build and install
Please install the implementation using a *conda* environment:
```bash
# Create conda environment
conda env create -f environment.yml

# Download and install
git clone https://github.com/Rosemeis/halkoGPU.git
cd halkoSVD
python setup.py build_ext --inplace
pip3 install -e .
```

You can now run the program with the `halko` command.

## Usage
Provide `halko` with the file-prefix of the PLINK files.
```
# Check help message of the program
halko -h

# GPU batch-based with batch-size of 8192 SNPs
halko --bfile input --threads 8 --n_eig 10 --batch 8192 --out halko.gpu

# CPU batch-based with batch-size of 8192 SNPs
halko --bfile input --threads 8 --n_eig 10 --batch 8192 --cpu --out halko.cpu

# CPU but loading all data into memory
halko --bfile input --threads 8 --n_eig 10 --full --out halko.cpu.full
```
