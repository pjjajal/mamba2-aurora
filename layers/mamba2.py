import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.ssd import ssd_minimal_discrete


class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        n_heads=2,
        n_groups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        conv_bias=True,
        chunk_size=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.n_groups = n_groups
        # compute d_inner based on the expand factor
        self.d_inner = d_model * expand
        self.branch_dim = self.d_inner  # we use this local variable for simplicity

        self.n_heads = n_heads
        assert (
            self.branch_dim % self.n_heads == 0
        ), "branch_dim must be divisible by n_heads"
        self.head_dim = self.branch_dim // self.n_heads
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size

        d_in_proj = (
            2 * self.branch_dim  # for z, x
            + 2 * self.n_groups * self.d_state  # for B, C
            + self.n_heads  # for dt
        )  # [z, x, B, C, dt]
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        # Convolution layers
        self.xBC_dim = self.branch_dim + 2 * self.n_groups * self.d_state
        self.conv_xBC = nn.Conv1d(
            in_channels=self.xBC_dim,
            out_channels=self.xBC_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.xBC_dim,
            padding=d_conv - 1,  # causal conv.
        )

        # Learnable init states
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.n_heads, self.head_dim, self.d_state)
            )
            self.init_states._no_weight_decay = True

        # Initialize log dt bias
        dt = torch.exp(
            torch.randn(self.n_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse dt
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Initialize A matrix
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D parameter
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True

        self.norm = nn.RMSNorm(self.d_inner)

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor):
        bs, seqlen, d_model = x.shape

        # Handling A and initial states
        A = -torch.exp(self.A_log)  # [n_heads]

        zxbcdt = self.in_proj(x)  # [B, seqlen, d_in_proj]
        z, xBC, dt = torch.split(
            zxbcdt, [self.branch_dim, self.xBC_dim, self.n_heads], dim=-1
        )  # [B, seqlen, branch_dim], [B, seqlen, xBC_dim], [B, seqlen, n_heads]
        dt = F.softplus(dt + self.dt_bias)  # [B, seqlen, n_heads]

        # Convolution
        xBC = F.silu(self.conv_xBC(xBC.transpose(1, 2))).transpose(1, 2)
        xBC = xBC[:, :seqlen, :]

        # Split xBC into x, B, C
        x, B, C = torch.split(
            xBC,
            [
                self.branch_dim,
                self.n_groups * self.d_state,
                self.n_groups * self.d_state,
            ],
            dim=-1,
        )

        # [B, seqlen, branch_dim] -> [B, seqlen, n_heads, head_dim]
        x = rearrange(x, "b l (h p) -> b l h p", p=self.head_dim).contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", g=self.n_groups).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", g=self.n_groups).contiguous()

        # SSM computation
        y, final_state = ssd_minimal_discrete(
            # We discretize x rather than B. The result is the same as ssd_chunk_scan_combined_ref.
            x * dt.unsqueeze(-1),
            A * dt,
            B,
            C,
            block_len=self.chunk_size,
            initial_states=(self.init_states if self.learnable_init_states else None),
        )
        y = y + x * self.D.unsqueeze(-1)  # (B, L, nheads, d_head)

        # [*, n_head, head_dim] -> [*,  branch_dim]
        y = rearrange(y, "b l h p -> b l (h p)")

        y = z * F.silu(y)
        y = self.norm(y)
        y = self.out_proj(y).contiguous()  # [*, d_inner] -> [*, d_model]
        return y
