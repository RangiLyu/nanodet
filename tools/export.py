import os
import torch
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

def main(config, model_path, output_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    dummy_input = torch.autograd.Variable(torch.randn(1, 3, input_shape[0], input_shape[1]))
    torch.onnx.export(model, dummy_input, output_path, verbose=True, keep_initializers_as_inputs=True, opset_version=11)
    print('finished exporting onnx ')

if __name__ == '__main__':
    cfg_path = r"config/nanodet-m.yml"
    model_path = r"nanodet_m.pth"
    out_path = r'output.onnx'
    load_config(cfg, cfg_path)
    main(cfg, model_path, out_path, input_shape=(320, 320))