from .sam_loss import SAMLoss
from .unet_loss import UNETloss

__all__ = ['SAMLoss', 'UNETloss']

def get_loss_fn(name: str, **kwargs):
    if name in __all__:
        loss_cls = globals().get(name, None)
        assert loss_cls is not None, f"Loss class {name} not found."
        return loss_cls(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")

