import torch

class LogSumExpSafe(torch.autograd.Function):
    """Implemented by Jason Eisner 2020 adapted by Brian Lu 2023.
    Implements a torch function that is exactly like logaddexp,
    but is willing to zero out nans on the backward pass.
    """

    @staticmethod
    def forward(ctx, input, dim):  # type: ignore
        with torch.enable_grad():
            output = torch.logsumexp(input, dim=dim)  # internal copy of output
        ctx.save_for_backward(input, output, torch.tensor(dim, dtype=torch.long))
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        input, output, dim = ctx.saved_tensors
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        grad_input, = torch.autograd.grad(output, input, grad_output)
        if input.requires_grad:
            zeros = grad_output.new_zeros(input.size())
            g1 = torch.where((grad_output == 0).unsqueeze(dim.item()).expand_as(grad_input), zeros, grad_input)
        else:
            g1 = None
        torch.set_anomaly_enabled(enabled)
        return g1, None

def logsumexp(x, dim) -> torch.Tensor:
    return LogSumExpSafe.apply(x, dim) # type:ignore

