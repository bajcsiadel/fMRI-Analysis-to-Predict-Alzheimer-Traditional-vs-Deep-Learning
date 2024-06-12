import dataclasses as dc
from pathlib import Path

from utils.errors.file_errors import UnsupportedExtensionError


@dc.dataclass
class DataConfig:
    location: Path
    metafile: Path

    def __setattr__(self, key, value):
        match key:
            case "location":
                if not value.exists():
                    raise ValueError("Data location does not exist")
                elif not value.is_dir():
                    raise ValueError("Data location is not a directory")
            case "metafile":
                if not value.exists():
                    if "location" in vars(self):
                        if not (self.location / value).exists():
                            raise ValueError("Data metafile does not exist")
                        else:
                            value = self.location / value
                    else:
                        raise ValueError("Data metafile does not exist")
                elif not value.is_file():
                    raise ValueError("Data metafile is not a file")
                elif value.suffix != ".csv":
                    raise UnsupportedExtensionError("Data metafile is not a CSV file")

        super().__setattr__(key, value)

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_data_type_validation", group=group, node=DataConfig)

        return config_store
