# Implement the RCPS framework from @zhou2020, with explicitly shared parameters
# and (relatively) intuitive layering for Sequential PyTorch model.
from einops import rearrange
import torch.nn as nn
import torch
import math


class RCPS(nn.Module):
    def __init__(self, in_kernels, out_kernels, kernel_size, padding="valid"):
        super().__init__()
        self.padding = padding
        self.in_kernels = in_kernels
        self.out_kernels = out_kernels
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(
            torch.empty(self.out_kernels, self.in_kernels, self.kernel_size),
            requires_grad=True,
        )
        self.c1d = lambda s, k: nn.functional.conv1d(s, k, padding=self.padding)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x):
        """Scan the output from previous RCPS layer, fwd on 1st half, rev on 2nd"""
        x1, x2 = rearrange(x, "b (split c) l -> split b c l", split=2)
        fwd_out = self.c1d(x1, self.weights)
        rev_out = self.c1d(x2, self.weights.flip(-2, -1)).flip(-2)
        return torch.cat([fwd_out, rev_out], dim=1)


class RCPSInput(RCPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_kernels=4, **kwargs)

    def forward(self, dna):
        """Scan dna with fwd and RC parameters, then concat along channel"""
        fwd_out = self.c1d(dna, self.weights)
        rev_out = self.c1d(dna, self.weights.flip(-2, -1)).flip(-2)
        return torch.cat([fwd_out, rev_out], dim=1)


class RCPSOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Separate, rc one, and combine fwd and rc outputs from previous layer"""
        x1, x2 = rearrange(x, "b (split c) l -> split b c l", split=2)
        return x1 + x2.flip(-1, -2)


class RCPSBatchNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.batch_norm = nn.BatchNorm1d(self.num_channels)

    def forward(self, x):
        """Normalise the fwd and rc channels in this batch separately"""
        _, _, orig_length = x.shape
        concat_x = torch.concat(
            [x[:, : self.num_channels, ...], x[:, self.num_channels :, ...].flip(-2)],
            dim=-1,
        )
        normed_x = self.batch_norm(concat_x)
        out_x = torch.concat(
            [normed_x[:, :, :orig_length], normed_x[:, :, orig_length:].flip(-2)], dim=1
        )
        return out_x


if __name__ == "__main__":
    # AAATTATCCGGCG: one-hot encoded, stored in (batch, channel, length) order
    fwd_seq = torch.Tensor(
        [
            [
                [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        ]
    )

    # Example sequential model
    example_model = nn.Sequential(
        RCPSInput(out_kernels=8, kernel_size=3, padding="same"),
        RCPS(8, 16, 3, padding="same"),
        RCPSBatchNorm(16),
        RCPS(16, 3, 3, padding="same"),
        RCPSBatchNorm(3),
        RCPSOutput(),
    )

    # Verify same output when run with forward or reverse complement sequence
    out_fwd = example_model(fwd_seq)
    out_rc = example_model(fwd_seq.flip(-1, -2))
    print(torch.isclose(out_fwd, out_rc, atol=1e-6).all())
