import torch
import torch.nn.functional as F


 # ! Used to calulate the mean squared error 
 # * MSE(x,y) = L = {l_1, l_2, ..., l_N} where l_i = (x_i - y_i)^2 and N is the batch size
 # * when reduction='sum' we MSE(x,y) - sum(l_i) for i=1 to N 
 # MSE is the same as squared L2 norm between each element in the input x and target y (x and y have the same shape)
 # x and y are tensors of arbitrary shapes with a total of N elements each
 # returns a tensor 
 # ? A tensor is a multi-dimensional matrix containing elements of a single data type
 # ? When a tensor is created with 'requires_grad=True', then torch.autograd records operations on them for automatic differentiation
def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

# ! Used to calculate the citerion negative log bernoulli
# Also known as binary cross-entropy loss or log loss
# This is equivalent to the negative log likelihood for a bernoulli distribution
# Bernoulli distribution is a special type of binomial distribution
# which takes value 1 with probability p, and value 0 with probability 1 - p
def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()

# ! Used to calculate the relative error
def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
