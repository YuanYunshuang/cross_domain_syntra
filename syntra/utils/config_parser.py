import sys, os
from hydra import compose, initialize_config_module
import hydra


@hydra.main()
def load_config(args):
    """Load a configuration file using Hydra and merge the overriding args."""
    cfg_file = args.config
    # read yaml file with hydra
    initialize_config_module(config_module="syntra.configs", version_base=None)
    cfg = compose(config_name=cfg_file, overrides=args.overrides)
    return cfg




