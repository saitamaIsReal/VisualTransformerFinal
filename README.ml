# FSII Analysis with ViT Image Classification

This repository demonstrates how to approximate 2nd-order Faithful Shapley Interaction Indices (FSII) on Vision Transformer (ViT) image classification models using [SHAPIQ](https://github.com/mmschlk/shapiq).

## Contents

- `vit_regFSII.ipynb`: Jupyter Notebook with step-by-step FSII analysis on a 7×7 patch grid.
- `vit_fsii_grid.py`: Standalone script to run FSII approximation on a fixed grid.
- `environment.yml`: Conda environment definition.
- `requirements.txt`: Pip requirements file.

## Environment Setup

### Using Conda

```bash
conda env create -f environment.yml
conda activate vit_fsii
jupyter lab     # or jupyter notebook
```

### Using Virtualenv + Pip

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .\.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab     # or jupyter notebook
```

## Usage

1. **Open the notebook**:
   ```bash
   jupyter lab vit_regFSII.ipynb
   ```
2. **Run all cells** to:
   - Load and display an example image with a 7×7 patch grid.
   - Mask selected patches and compute model logits.
   - Approximate FSII scores and visualize the top-5 interacting patch pairs.

3. **Standalone script**:
   ```bash
   python vit_fsii_grid.py
   ```
   This will print top-5 pairwise FSII values and show the result image.

## Notes

- We apply a *monkey-patch* to disable SHAPIQ’s border-trick, avoiding dimension errors on large budgets.
- `sampling_weights` controls the fraction of budget allocated to single patches vs. patch pairs.
- `return_interactions=False` suppresses storing raw samples and speeds up final runs.


