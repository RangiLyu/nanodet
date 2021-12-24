import tempfile

from torch.utils.tensorboard import SummaryWriter

from nanodet.util import NanoDetLightningLogger, cfg, load_config


def test_logger():
    tmp_dir = tempfile.TemporaryDirectory()
    logger = NanoDetLightningLogger(tmp_dir.name)

    writer = logger.experiment
    assert isinstance(writer, SummaryWriter)

    logger.info("test")

    logger.log_hyperparams({"lr": 1})

    logger.log_metrics({"mAP": 30.1}, 1)

    load_config(cfg, "./config/legacy_v0.x_configs/nanodet-m.yml")
    logger.dump_cfg(cfg)

    logger.finalize(None)
