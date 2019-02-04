import torch.nn.functional as F


_loss_name_2_fn = {
    # TODO add more later
    "cross_entropy": F.cross_entropy,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
    "binary_cross_entropy": F.binary_cross_entropy,
    "nll_loss": F.nll_loss
}


def get_loss_fn_maps():
    return _loss_name_2_fn
