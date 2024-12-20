import numpy as np
import torch


def collate_padded_tuples(batch):
    batch = tuple(zip(*batch))
    lengths = [[x.shape[1] for x in inputs] for inputs in batch]
    max_lengths = [max(x) for x in lengths]
    batch = [
        np.array([np.pad(x, [(0, 0), (0, max(0, max_len - x.shape[1]))]) for x in inputs])
        for inputs, max_len in zip(batch, max_lengths)
    ]
    return tuple(
        (torch.as_tensor(x), torch.as_tensor(ln)) for x, ln in zip(batch, lengths)
    )


def markdown_format_code(text):
    return "    " + text.replace("\n", "\n    ")
