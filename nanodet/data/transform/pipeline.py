from .warp import warp_and_resize
from .color import color_aug_and_norm
import functools


class Pipeline:
    def __init__(self,
                 cfg,
                 keep_ratio):
        self.warp = functools.partial(warp_and_resize,
                                      warp_kwargs=cfg,
                                      keep_ratio=keep_ratio)
        self.color = functools.partial(color_aug_and_norm,
                                       kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta=meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta
