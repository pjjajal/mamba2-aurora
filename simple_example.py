import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mamba2 import Mamba2Block


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def main():
    device = get_device()
    print("Using device:", device)

    mamba2_blk = Mamba2Block(
        d_model=256,
        d_state=64,
        chunk_size=32, # this is just for the chunkwise parallel block.
    )
    mamba2_blk = mamba2_blk.to(device)

    batch_size = 8
    seq_len = 512

    x = torch.randn(batch_size, seq_len, 256, device=device)

    y = mamba2_blk(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()