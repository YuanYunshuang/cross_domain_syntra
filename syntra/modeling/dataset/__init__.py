from .smol import SMOLDataset
from .syntra import SynTraDataset
from .cd_syntra import CrossDomainSynTraDataset
from .cd_syntra_efficient import CrossDomainSynTraDatasetE


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "smol":
        return SMOLDataset(**kwargs)
    elif dataset_name == "syntra":
        return SynTraDataset(**kwargs)
    elif dataset_name == "cd_syntra":
        return CrossDomainSynTraDataset(**kwargs)
    elif dataset_name == "cd_syntra_efficient":
        return CrossDomainSynTraDatasetE(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")