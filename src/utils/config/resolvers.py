import os
import pathlib

from hydra.types import RunMode
from omegaconf import omegaconf

from utils.environment import get_env

log_created = {}


def _get_package_name():
    import traceback

    source_location = pathlib.Path(get_env("SOURCE_LOCATION"))

    stack = traceback.extract_stack()
    i = 0
    source_file = None
    while i < len(stack) and source_file is None:
        current_file = pathlib.Path(stack[i].filename)
        if current_file.is_relative_to(source_location):
            source_file = current_file
        i += 1

    package_location = source_file.relative_to(source_location).with_suffix("")
    return ".".join(package_location.parts)


def _get_results_location():
    return pathlib.Path(get_env("RESULTS_LOCATION"), _get_package_name())


def _create(run_mode, sweep_dir, filename="progress.log"):
    if run_mode == RunMode.RUN:
        # do not create progress file
        return os.devnull
    if type(sweep_dir) is not pathlib.Path:
        sweep_dir = pathlib.Path(sweep_dir)

    filename = sweep_dir / filename

    global log_created

    if filename not in log_created or not log_created[filename]:
        log_created[filename] = True
        filename.parent.mkdir(parents=True, exist_ok=True)
    return filename


def resolve_results_location():
    omegaconf.OmegaConf.register_new_resolver("results_location", _get_results_location)


def resolve_package_name():
    omegaconf.OmegaConf.register_new_resolver("package_name", _get_package_name)


def resolve_create():
    omegaconf.OmegaConf.register_new_resolver("create", _create)


def add_all_custom_resolvers():
    resolve_create()
    resolve_package_name()
    resolve_results_location()
