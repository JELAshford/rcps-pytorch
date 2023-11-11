# RCPS (Reverse-Complement Parameter Sharing)

`RCPS` layers for machine learning on DNA sequences in [PyTorch](https://pytorch.org/), based on [the fantastic work by Zhou et. al.](https://www.biorxiv.org/content/10.1101/2020.11.04.368803v2). This version of `RCPS` acts as a near drop-in replacement for PyTorch's standard `Conv1d` layers with the exception that additional `RCPS-`type layers must be inserted before and after the convolutions to ensure the extra channels get handled properly. 

The benefit of this version of `RCPS` layers is that they only store one version of the learned kernel parameters while still computing the forward and reverse convolutions on an input, emitting both outputs in a single channel-mirrored array. Additionally, `RCPSBatchNorm` provides a convenient and encoding-safe way to normalise within an `RCPS` block. For full details, please see the paper linked above.

## Usage

RCPS layers can be used to carry out convolutions on DNA sequences and downstream encodings from previous RCPS layers. The easiest way of using them is as part of a torch `Sequential` object, which allows you to easily chain the `RCPS` layers between the needed `RCPSInput` and `RCPSOutput` layers. 

To begin with, we can generate some one-hot encoded DNA sequence to run through our model. Usually I would use the encoder provided in the excellent [`enformer_pytorch`](https://github.com/lucidrains/enformer-pytorch) repository, but I have hard-coded one here to avoid an extra dependency. 

```python
from rcps import RCPSInput, RCPS, RCPSBatchNorm, RCPSOutput
import torch.nn as nn
import torch

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
```

We can use a torch `Sequential` object to create our model, ensuring to wrap all of our `RCPS`-layer logic with the -`Input` and -`Output` layers.

```python
example_model = nn.Sequential(
    RCPSInput(out_kernels=8, kernel_size=3, padding="same"),
    RCPS(8, 16, 3, padding="same"),
    RCPSBatchNorm(16),
    RCPS(16, 3, 3, padding="same"),
    RCPSBatchNorm(3),
    RCPSOutput(),
)
```

Finally, we can verify that the model performs the same on both the forward and reverse-complement version of the input. Here, reverse-complementing the one-hot encoded DNA string is performed by 'flipping' both the channel and length axis. 

```python
out_fwd = example_model(fwd_seq)
out_rc = example_model(fwd_seq.flip(-1, -2))
print(torch.isclose(out_fwd, out_rc, atol=1e-6).all())
# tensor(True)
```
## Limitations
- Currently, the nominal way to use the `RCPSInput` is with an input that represents DNA sequences one-hot encoded, in the shape `(batch, 4, length)`. In theory, there could be other encodings for DNA that do not use 4 channels but do still benefit from the easy 'reverse complement' action of flipping all not-batch dimensions. 
- Due to PyTorch being unable to reverse index (i.e. `flip`) without a copy, multiple copies of the input and weights are created during forward passes. I would welcome a way to change this, but unfortunately I don't know a way to do that without moving away from PyTorch.


## Future Work
- [ ] Find ways that the RCPS can be more smoothly integrated into `Sequential` models without the extra `RCPSInput` and `RCPSOutput` layers. (Pull Requests or Discussions welcome!)
- [ ] Benchmark the performance of the RCPS technique against other reverse-complement preserving encodings/layers.