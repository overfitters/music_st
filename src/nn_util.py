from torch import nn


class RNNWrapper(nn.Module):
    """Wrapper for plugging an RNN into a CNN."""

    def __init__(self, rnn=None, return_state=False, return_output=None):
        super().__init__()
        if rnn is not None:
            self.rnn = rnn
        else:
            self.rnn = nn.GRU(input_size=1024, hidden_size=1024)
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

    def forward(self, input):
        output = self.module(input)
        if output.shape != input.shape:
            raise RuntimeError(
                f"Expected output to have shape {input.shape}, got {output.shape}"
            )
        return output + input
