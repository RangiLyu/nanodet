import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nanodet.data.collate import collate_function
from nanodet.model.arch import build_model
from nanodet.trainer import build_trainer
from nanodet.util import Logger, cfg, load_config


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        data = {
            "img": torch.randn((3, 32, 32)),
            "img_info": {
                "file_name": "dummy_data.jpg",
                "height": 500,
                "width": 500,
                "id": 1,
            },
            "gt_bboxes": np.array(
                [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
            ),
            "gt_labels": np.array([1, 2]),
            "warp_matrix": [np.eye(3), np.eye(3)],
        }
        return data


class DummyEvaluator:
    metric_names = ["mAP"]

    def evaluate(self, results, save_dir, rank=-1):
        return {"mAP": 0.5}


def test_trainer():
    tmp_dir = tempfile.TemporaryDirectory()

    load_config(cfg, "./config/nanodet-m.yml")
    cfg.defrost()
    cfg.model.arch.backbone.pretrain = False
    cfg.schedule.total_epochs = 4
    cfg.schedule.val_intervals = 1
    cfg.schedule.warmup.steps = 2
    cfg.save_dir = tmp_dir.name
    dummy_dataset = DummyDataset()
    train_loader = DataLoader(
        dummy_dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_function,
        drop_last=True,
    )
    val_loader = DataLoader(
        dummy_dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_function,
        shuffle=False,
        drop_last=False,
    )

    model = build_model(cfg.model)
    logger = Logger(-1, tmp_dir.name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    trainer = build_trainer(rank=-1, cfg=cfg, model=model, logger=logger, device=device)
    trainer.run(train_loader, val_loader, DummyEvaluator())
