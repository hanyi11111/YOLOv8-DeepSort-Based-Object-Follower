# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig, OmegaConf

from ultralytics.yolo.configs.hydra_patch import check_config_mismatch


def get_config(config: Union[str, DictConfig], overrides: Union[str, Dict] = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        config (Union[str, DictConfig]): Configuration data in the form of a file name or a DictConfig object.
        overrides (Union[str, Dict], optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        OmegaConf.Namespace: Training arguments namespace.
    """
    if overrides is None:
        overrides = {}
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    elif isinstance(config, Dict):
        config = OmegaConf.create(config)
    # override
    if isinstance(overrides, str):
        overrides = OmegaConf.load(overrides)
    elif isinstance(overrides, Dict):
        overrides = OmegaConf.create(overrides)

# check_config_mismatch(dict(overrides).keys(), dict(config).keys())

    # PATCH_OMEGA_CLASSES_MERGE: merge 需要 overrides 含 classes，否则 Missing key classes
    if "classes" not in overrides:
        overrides = OmegaConf.merge(overrides, OmegaConf.create({"classes": []}))
    return OmegaConf.merge(config, overrides)
