import dataclasses as dc
from pathlib import Path

from utils.errors.file_errors import UnsupportedExtensionError


@dc.dataclass
class CSVFileConfig:
    filename: Path
    parameters: dict
    must_exist: bool


@dc.dataclass
class DataConfig:
    location: Path
    metadata: CSVFileConfig
    selected_patients: CSVFileConfig

    def __setattr__(self, key, value):
        match key:
            case "location":
                if not value.exists():
                    raise ValueError(f"Data location {value} does not exist")
                elif not value.is_dir():
                    raise ValueError(f"Data location {value} is not a directory")
            case "metadata" | "selected_patients":
                if not value.filename.is_absolute() and "location" in vars(self):
                    value.filename = self.location / value.filename
                if value.must_exist:
                    if not value.filename.exists():
                        raise ValueError(f"CSV file {key!r} does not exist")
                    elif not value.filename.is_file():
                        raise ValueError(f"{key!r} is not a file")
                if value.filename.suffix != ".csv":
                    raise UnsupportedExtensionError("Data metafile is not a CSV file")
        super().__setattr__(key, value)

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_data_type_validation", group=group, node=DataConfig)

        return config_store
