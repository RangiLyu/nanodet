from .one_stage import OneStage


class GFL(OneStage):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg,
                 head_cfg, ):
        super(GFL, self).__init__(backbone_cfg,
                                   fpn_cfg,
                                   head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x
