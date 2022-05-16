import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union, Any

import pandas as pd
from ruamel.yaml import YAML

from neuralhydrology.utils.baseconfig import BaseConfig


class AssimilationConfig(BaseConfig):
    """Configuration class for data assimilation specific arguments.

    Parameters
    ----------
    yml_path_or_dict : Union[Path, dict]
        Either a path to the config file or a dictionary of configuration values.
    dev_mode : bool, optional
        If dev_mode is off, the config creation will fail if there are unrecognized keys in the passed config
        specification. dev_mode can be activated either through this parameter or by setting ``dev_mode: True``
        in `yml_path_or_dict`.

    Raises
    ------
    ValueError
        If the passed configuration specification is neither a Path nor a dict or if `dev_mode` is off (default) and
        the config file or dict contain unrecognized keys.
    """

    # Lists of deprecated config keys and purely informational metadata keys, needed when checking for unrecognized
    # config keys since these keys are not properties of the Config class.
    _deprecated_keys = []
    _metadata_keys = []

    def __init__(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        super(AssimilationConfig, self).__init__(yml_path_or_dict=yml_path_or_dict, dev_mode=dev_mode)

        # TODO: Check if I can put this method back into BaseConfig
        if not (self._cfg.get('dev_mode', False) or dev_mode):
            self._check_cfg_keys(self._cfg)

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        """Checks the config for unknown keys. """
        property_names = [p for p in dir(AssimilationConfig) if isinstance(getattr(AssimilationConfig, p), property)]

        unknown_keys = [
            k for k in cfg.keys() if k not in property_names and k not in AssimilationConfig._deprecated_keys and
            k not in AssimilationConfig._metadata_keys
        ]
        if unknown_keys:
            raise ValueError(f'{unknown_keys} are not recognized config keys.')

    @property
    def assimilation_lead_time(self) -> int:
        return self._get_value_verbose("assimilation_lead_time")

    @property
    def assimilation_targets(self) -> List[str]:
        assimilation_targets = self._as_default_list(self._cfg.get("assimilation_targets", []))
        if not assimilation_targets:
            raise ValueError("At least one assimilation target has to be specified")
        return assimilation_targets

    @property
    def assimilation_window(self) -> int:
        return self._get_value_verbose("assimilation_window")

    @property
    def epochs(self) -> int:
        return self._cfg.get("epochs", 200)

    @property
    def history(self) -> int:
        return self._get_value_verbose("history")

    @property
    def learning_rate(self) -> Dict[int, float]:
        if ("learning_rate" in self._cfg.keys()) and (self._cfg["learning_rate"] is not None):
            if isinstance(self._cfg["learning_rate"], float):
                return {0: self._cfg["learning_rate"]}
            elif isinstance(self._cfg["learning_rate"], dict):
                return self._cfg["learning_rate"]
            else:
                raise ValueError("Unsupported data type for learning rate. Use either dict (epoch to float) or float.")
        else:
            raise ValueError("No learning rate specified in the config (.yml).")

    @property
    def learning_rate_drop_factor(self) -> float:
        return self._cfg.get("learning_rate_drop_factor", 0.9)

    @property
    def learning_rate_epoch_drop(self) -> int:
        return self._cfg.get("learning_rate_epoch_drop", 5)

    @property
    def loss(self) -> str:
        return self._get_value_verbose("loss")

    @property
    def model_dropout(self) -> bool:
        model_dropout = self._cfg.get("model_dropout", False)
        if model_dropout is None:
            return False
        else:
            return model_dropout

    @property  # required by loss class
    def no_loss_frequencies(self) -> List[str]:
        return self._as_default_list(self._cfg.get("no_loss_frequencies", []))

    @property
    def optimizer(self) -> str:
        return self._get_value_verbose("optimizer")

    @property  # required by loss class
    def predict_last_n(self) -> int:
        return self._get_value_verbose("predict_last_n")

    @property
    def regularization(self) -> List[str]:
        return self._as_default_list(self._cfg.get("regularization", []))

    @property
    def seq_length(self) -> int:
        return self._get_value_verbose("seq_length")

    @property  # required by loss class
    def target_loss_weights(self) -> List[float]:
        return self._cfg.get("target_loss_weights", None)

    @property
    def timestep_dropout(self) -> float:
        timestep_dropout = self._cfg.get("timestep_dropout", 0.0)
        if timestep_dropout is None:
            return 0.0
        else:
            if timestep_dropout >= 1.0:
                raise ValueError("'timestep_dropout' has to be smaller than 1.0 (and larger or equal to 0.0)")
            else:
                return timestep_dropout

    @property  # required by loss class
    def target_variables(self) -> List[str]:
        return self._get_value_verbose("target_variables")
