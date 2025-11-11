from .smol import SMOLDataset
from .syntra import SynTraDataset
from .cd_syntra import CrossDomainSynTraDataset


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "smol":
        return SMOLDataset(**kwargs)
    elif dataset_name == "syntra":
        return SynTraDataset(**kwargs)
    elif dataset_name == "cd_syntra":
        return CrossDomainSynTraDataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")