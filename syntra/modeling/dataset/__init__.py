from .smol import SMOLDataset
from .syntra import SynTraDataset
from .cd_syntra import CrossDomainSynTraDataset
from .cd_syntra_efficient import CrossDomainSynTraDatasetE
from .cd_syntra2_efficient import CrossDomainSynTraDataset2E
from .unet_dataset import UNetDataset


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "smol":
        return SMOLDataset(**kwargs)
    elif dataset_name == "syntra":
        return SynTraDataset(**kwargs)
    elif dataset_name == "cd_syntra":
        return CrossDomainSynTraDataset(**kwargs)
    elif dataset_name == "cd_syntra_efficient":
        return CrossDomainSynTraDatasetE(**kwargs)
    elif dataset_name == "cd_syntra2_efficient":
        return CrossDomainSynTraDataset2E(**kwargs)
    elif dataset_name == "unet":
        return UNetDataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")