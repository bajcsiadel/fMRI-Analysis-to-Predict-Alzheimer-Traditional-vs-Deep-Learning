import logging
import os
import sys
import traceback
import typing as typ
from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter


class BasicLogger(logging.Logger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :type name: str
    """

    def __init__(self, name):
        super().__init__(name)

        if type(HydraConfig.get().verbose) is bool or (
                type(HydraConfig.get().verbose) is str and
                HydraConfig.get().verbose.lower() == name.lower()
        ):
            # set debug level
            self.setLevel(logging.DEBUG)

        self.parent = logging.root

        self._log_dir = Path(HydraConfig.get().runtime.output_dir)
        self.info(f"Logging to {self._log_dir}")

        # Ensure the directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.getLogger().manager.loggerDict[name] = self

    @property
    def log_dir(self):
        return self._log_dir

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        """
        Write a message to the log file

        :param level: the level of the message, e.g. logging.INFO
        :type level: int
        :param msg: the message string to be written to the log file
        :type msg: str
        :param args: arguments for the message
        :type args:
        :param exc_info: exception info. Defaults to None
        :param extra: extra information. Defaults to None
        :type extra: typ.Mapping[str, object] | None
        :param stack_info: whether to include stack info. Defaults to False
        :type stack_info: bool
        :param stacklevel: the stack level. Defaults to 1
        :type stacklevel: int
        """
        if type(msg) is not str:
            msg = str(msg)
        lines = msg.splitlines()
        indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
        lines[0] = lines[0].strip()
        for line in lines:
            super()._log(
                level,
                f"{indent}{line}",
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,
            )

    def exception(self, ex, warn_only=False, **kwargs):
        """
        Customize logging an exception

        :param ex:
        :type ex: Exception
        :param warn_only: Defaults to False
        :type warn_only: bool
        :type ex: Exception
        """
        if warn_only:
            log_fn = self.warning
        else:
            log_fn = self.error
        log_fn(f"{type(ex).__name__}: {ex}", **kwargs)
        log_fn(traceback.format_exc(), **kwargs)

    def __call__(self, message):
        """
        Log a message

        :param message:
        :type message: str
        """
        level, msg = message.split(": ", 1)
        match level:
            case "INFO":
                self.info(msg)
            case "WARNING":
                self.warning(msg)
            case "ERROR":
                self.error(msg)
            case _:
                self.log(logging.INFO, message)


class TrainLogger(BasicLogger):
    """
    Object for managing the log directory during training

    :param name: The name of the logger
    :type name: str
    :param output_dirs: The output directories
    :type output_dirs: utils.config.output.OutputConfig
    """

    def __init__(self, name, output_dirs):
        super().__init__(name)

        self.__output_dirs = output_dirs

        # Ensure the directories exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self._tensorboard_writer = SummaryWriter(str(self.tensorboard_dir))

    @property
    def checkpoint_dir(self):
        return self._log_dir / self.__output_dirs.checkpoints_dir

    @property
    def metadata_dir(self):
        return self._log_dir / self.__output_dirs.metadata_dir

    @property
    def tensorboard_dir(self):
        return self._log_dir / self.__output_dirs.tensorboard_dir

    @property
    def tensorboard(self):
        return self._tensorboard_writer

    def log_command_line(self):
        """
        Generate a script that can be used to run the experiment again
        """
        python_file = sys.argv[0]
        params = " ".join(HydraConfig.get().overrides.task)
        screen_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        bash_script_file = self.metadata_dir / "run-experiment.sh"
        with bash_script_file.open(mode="w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("\n")
            fd.write(f"cd {os.getenv('SOURCE_LOCATION')}\n")
            fd.write("\n")
            fd.write("# check if environment exists\n")
            fd.write("poetry env list > /dev/null\n")
            fd.write('if { [ $? -ne 0 ] && [ -f "pyproject.toml" ]; }\n')
            fd.write("then\n")
            fd.write("\tpoetry install\n")
            fd.write("else\n")
            fd.write(
                '\techo "No pyproject.toml found. '
                'Please run this script from the project root."\n'
            )
            fd.write("\texit 1\n")
            fd.write("fi\n")
            fd.write("\n")
            fd.write(f"screen -dmS {screen_name}\n")
            fd.write(
                f"screen -S {screen_name} -X stuff "
                f'"poetry run python {python_file} {params}"\n'
            )
            fd.write("# attaching the screen\n")
            fd.write(f"screen -r {screen_name}\n")

    def __del__(self):
        self._tensorboard_writer.close()
