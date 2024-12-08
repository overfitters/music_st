from torch import nn


class RNNWrapper(nn.Module):
    """Wrapper for plugging an RNN into a CNN."""

    def __init__(self, rnn, return_state=False, return_output=None):
        super().__init__()
        self.rnn = rnn
        self.return_state = return_state
        self.return_output = return_output or not return_state

    def forward(self, inp):
        output, state = self.rnn(inp.transpose(1, 2))
        output = output.transpose(1, 2)
        state = state.transpose(0, 1).reshape(inp.shape[0], -1)
        if self.return_output and self.return_state:
            return output, state
        if self.return_state:
            return state
        return output


class ResidualWrapper(nn.Module):
    """Wrapper for adding a skip connection around a module."""

    def __init__(self, module=None):
        super().__init__()
        if module is not None:
            self.module = module

    def forward(self, inp):
        output = self.module(inp)
        if output.shape != inp.shape:
            raise RuntimeError(
                f"Expected output to have shape {inp.shape}, got {output.shape}"
            )
        return output + inp
