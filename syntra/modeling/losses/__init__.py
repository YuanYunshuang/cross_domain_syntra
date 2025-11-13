from .sam_loss import SAMLoss

__all__ = ['SAMLoss']

def get_loss_fn(name: str, **kwargs):
    if name in __all__:
        loss_cls = globals().get('SAMLoss', None)
        assert loss_cls is not None, f"Loss class {name} not found."
        return loss_cls(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")

