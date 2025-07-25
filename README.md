# Reimplementation of SRGAN & SRCNN

This repository contains a reimplementation of the Super-Resolution Generative Adversarial Network (SRGAN) and the Super-Resolution Convolutional Neural Network (SRCNN) using PyTorch.

# Using This Repo

1. Clone the repository:

```bash
git clone https://github.com/keanteng/super-resolution
```

2. LFS Error Fixing

```bash
# Create a new branch without the problematic file
git checkout --orphan new-main

# Add all files except the large one
git add .
git reset slides-2/presentation-version-2.pdf

# Commit
git commit -m "Clean repository without large files"

# Replace main branch
git branch -D main
git branch -m new-main main

# Force push
git push origin main --force
```