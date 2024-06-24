import hydra
from omegaconf import omegaconf

from utils.config.resolvers import resolve_results_location
from utils.config.train import TrainConfig
from utils.environment import get_env
from utils.logger import TrainLogger


def run(cfg: TrainConfig, logger: TrainLogger):
    hydra.utils.instantiate(cfg.model.trainer, cfg, logger)


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIGURATIONS_LOCATION"),
    config_name="scripts_classify_fmri"
)
def main(cfg: TrainConfig):
    logger = TrainLogger(__name__, cfg.out_dirs)
    try:
        cfg = omegaconf.OmegaConf.to_object(cfg)

        run(cfg, logger)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    TrainConfig.add_type_validation()

    resolve_results_location()

    main()
