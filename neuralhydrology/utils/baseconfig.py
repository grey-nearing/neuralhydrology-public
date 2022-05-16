from collections import OrderedDict
from pathlib import Path
from typing import Union, Any

import pandas as pd
from ruamel.yaml import YAML


class BaseConfig(object):
    """Abstract Config class.

    Do not use this class for model training/evaluation. Instead use the `Config` class

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
        if isinstance(yml_path_or_dict, Path):
            self._cfg = self._read_and_parse_config(yml_path=yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            self._cfg = BaseConfig._parse_config(yml_path_or_dict)
        else:
            raise ValueError(f'Cannot create a config from input of type {type(yml_path_or_dict)}.')

    def as_dict(self) -> dict:
        """Return run configuration as dictionary.
        
        Returns
        -------
        dict
            The run configuration, as defined in the .yml file.
        """
        return self._cfg

    def dump_config(self, folder: Path, filename: str = 'config.yml'):
        """Save the run configuration as a .yml file to disk.

        Parameters
        ----------
        folder : Path
            Folder in which the configuration will be stored.
        filename : str, optional
            Name of the file that will be stored. Default: 'config.yml'.

        Raises
        ------
        FileExistsError
            If the specified folder already contains a file named `filename`.
        """
        yml_path = folder / filename
        if not yml_path.exists():
            with yml_path.open('w') as fp:
                temp_cfg = {}
                for key, val in self._cfg.items():
                    if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(str(elem))
                            temp_cfg[key] = temp_list
                        else:
                            temp_cfg[key] = str(val)
                    elif key.endswith('_date'):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(elem.strftime(format="%d/%m/%Y"))
                            temp_cfg[key] = temp_list
                        else:
                            # Ignore None's due to e.g. using a per_basin_period_file
                            if isinstance(val, pd.Timestamp):
                                temp_cfg[key] = val.strftime(format="%d/%m/%Y")
                    else:
                        temp_cfg[key] = val

                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            raise FileExistsError(yml_path)

    def update_config(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        """Update config arguments.
        
        Useful e.g. in the context of fine-tuning or when continuing to train from a checkpoint to adapt for example the
        learning rate, train basin files or anything else.
        
        Parameters
        ----------
        yml_path_or_dict : Union[Path, dict]
            Either a path to the new config file or a dictionary of configuration values. Each argument specified in
            this file will overwrite the existing config argument.
        dev_mode : bool, optional
            If dev_mode is off, the config creation will fail if there are unrecognized keys in the passed config
            specification. dev_mode can be activated either through this parameter or by setting ``dev_mode: True``
            in `yml_path_or_dict`.

        Raises
        ------
        ValueError
            If the passed configuration specification is neither a Path nor a dict, or if `dev_mode` is off (default)
            and the config file or dict contain unrecognized keys.
        """
        new_config = BaseConfig(yml_path_or_dict, dev_mode=dev_mode)

        self._cfg.update(new_config.as_dict())

    def _get_value_verbose(self, key: str) -> Union[float, int, str, list, dict, Path, pd.Timestamp]:
        """Use this function internally to return attributes of the config that are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key]

    @staticmethod
    def _as_default_list(value: Any) -> list:
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    def _as_default_dict(value: Any) -> dict:
        if value is None:
            return {}
        elif isinstance(value, dict):
            return value
        else:
            raise RuntimeError(f"Incompatible type {type(value)}. Expected `dict` or `None`.")

    @staticmethod
    def _parse_config(cfg: dict) -> dict:
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Add more config parsing if necessary
        return cfg

    def _read_and_parse_config(self, yml_path: Path):
        if yml_path.exists():
            with yml_path.open('r') as fp:
                yaml = YAML(typ="safe")
                cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(yml_path)

        cfg = self._parse_config(cfg)

        return cfg
