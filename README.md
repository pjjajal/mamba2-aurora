# Mamba2-Aurora

This repository provides a minimal, pure PyTorch implementation of the **Mamba2 architecture**, including both the Mamba2 block and the SSD algorithm.

- The Mamba2 block is implemented in [`layers/mamba2.py`](./layers/mamba2.py). It is a refactored version of the [original implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py), with improvements for clarity and modularity.
- All Triton dependencies have been removed. The SSD algorithm is implemented entirely in PyTorch and can be found in [`layers/ssd.py`](./layers/ssd.py).
