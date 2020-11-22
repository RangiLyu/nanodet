# NanoDet Config File Analysis

NanoDet using [yacs](https://github.com/rbgirshick/yacs) to read yaml config file.

## Saving path

```yaml
save_dir: PATH_TO_SAVE
```
Change save_dir to where you want to save logs and models. If path not exist, NanoDet will create it.

## Model

```yaml
model:
    arch:
        name: xxx
        backbone: xxx
        fpn: xxx
        head: xxx
```

Most detection model architecture can be devided into 3 parts: backbone, task head and connector between them(e.g. FPN, BiFPN, PAN...).

### Backbone

```yaml
backbone:
    name: ShuffleNetV2
    model_size: 1.0x
    out_stages: [2,3,4]
    activation: LeakyReLU
    with_last_conv: False
```

NanoDet using ShuffleNetV2 as backbone. You can modify model size, output feature levels and activation function. Moreover, NanoDet provides other lightweight backbones like **GhostNet** and **MobileNetV2**. You can also add your backbone network by importing it in `nanodet/model/backbone/__init__.py`.

### FPN

```yaml
fpn:
    name: PAN
    in_channels: [116, 232, 464]
    out_channels: 96
    start_level: 0
    num_outs: 3
```

NanoDet using modified [PAN](http://arxiv.org/abs/1803.01534) (replace downsample convs with interpolation to reduce amount of computations).

`in_channels` : a list of feature map channels extracted from backbone. 

`out_channels` : out put feature map channel.

### Head

```yaml
head:
    name: NanoDetHead
    num_classes: 80
    input_channel: 96
    feat_channels: 96
    stacked_convs: 2
    share_cls_reg: True
    octave_base_scale: 5
    scales_per_octave: 1
    anchor_ratios: [1.0]
    anchor_strides: [8, 16, 32]
    target_means: [.0, .0, .0, .0]
    target_stds: [0.1, 0.1, 0.2, 0.2]
    reg_max: 7
    norm_cfg:
      type: BN
    loss:
```

`name`: Task head class name

`num_classes`: number of classes

`input_channel`: input feature map channel

`feat_channels`: channel of task head convs

`stacked_convs`: how many conv blocks use in one task head

`share_cls_reg`: use same conv blocks for classification and box regression

***

**Anchor Setting**

Notice: NanoDet is a FCOS-style anchor free model based on Generialized Focal Loss. Anchor free means there is only one square base anchor on each feature map pixel which used for sampling.

`octave_base_scale`: base anchor scale

`scales_per_octave`: anchor scale

`anchor_ratios`: anchor ratio

`anchor_strides`: down sample stride of each feature map level

`target_means`: 

`target_stds`: 

`reg_max`: max value of per-level l-r-t-b distance

***

`norm_cfg`: normalization layer setting

`loss`: adjust loss functions and weights

## Data

