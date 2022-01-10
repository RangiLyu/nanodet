import tempfile

from torch.utils.tensorboard import SummaryWriter

from nanodet.util import NanoDetLightningLogger, NanoDetWandbLogger, cfg, load_config


def test_logger():
    tmp_dir = tempfile.TemporaryDirectory()
    logger = NanoDetLightningLogger(tmp_dir.name)
    wandb_logger = NanoDetWandbLogger(save_dir=tmp_dir.name, project="ci_test", anonymous="must")
    writer = logger.experiment
    assert isinstance(writer, SummaryWriter)

    logger.info("test")
    
    logger.log_hyperparams({"lr": 1})
    wandb_logger.log_hyperparams({"lr": 1})
    
    logger.log_metrics({"mAP": 30.1}, 1)
    wandb_logger.log_hyperparams({"mAP": 30.1})
    
    load_config(cfg, "./config/legacy_v0.x_configs/nanodet-m.yml")
    logger.dump_cfg(cfg)

    logger.finalize(None)
    wandb_logger.finalize(None)
