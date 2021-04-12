import os
import argparse
import torch
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


def main(config, model_path: str, output_path: str, input_shape=(320, 320)):
    logger = Logger(local_rank=-1, save_dir=config.save_dir, use_tensorboard=False)

    # Create model and load weights
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)

    # Convert backbone weights for RepVGG models
    if config.model.arch.backbone.name == 'RepVGG':
        deploy_config = config.model
        deploy_config.arch.backbone.update({'deploy': True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert
        model = repvgg_det_model_convert(model, deploy_model)

    # TorchScript: tracing the model with dummy inputs
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])  # Batch size = 1
        model.eval().cpu()
        model_traced = torch.jit.trace(model, example_inputs=dummy_input).eval()
        model_traced.save(output_path)
        print('Finished export to TorchScript')


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Convert .pth model weights to TorchScript.')
    parser.add_argument('--cfg_path',
                        type=str,
                        help='Path to .yml config file.')
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='Path to .ckpt model.')
    parser.add_argument('--out_path',
                        type=str,
                        default='nanodet.torchscript.pth',
                        help='TorchScript model output path.')
    parser.add_argument('--input_shape',
                        type=str,
                        default=None,
                        help='Model input shape.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(',')))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")
    main(cfg, model_path, out_path, input_shape)
    print("Model saved to:", out_path)
