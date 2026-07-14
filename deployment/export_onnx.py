# Copyright (c) OpenMMLab. All rights reserved.
"""Export an mmpretrain classification model to ONNX.

The exported graph takes a normalized float32 tensor of shape (N, 3, H, W)
(i.e. after resize/crop, BGR->RGB conversion and mean/std normalization) and
outputs the raw classification logits of shape (N, num_classes). Softmax and
preprocessing are kept outside of the graph, matching ``mode='tensor'`` of
mmpretrain classifiers.

Example:
    python export_onnx.py \
        ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
        --checkpoint ../mmpretrain/weights/resnet18_8xb32_in1k.pth \
        --output resnet18.onnx --dynamic-batch
"""
import argparse
import os.path as osp

import torch
import torch.nn as nn

from deploy_utils import (build_model, check_custom_ops, format_custom_ops,
                          get_input_shape, print_onnx_io, switch_to_deploy)


class ExportWrapper(nn.Module):
    """Wrap a classifier so that forward returns logits (mode='tensor')."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, mode='tensor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export an mmpretrain model to ONNX.')
    parser.add_argument(
        'config',
        help='Path to a config file (.py) or a model name in the ModelHub, '
        'e.g. resnet18_8xb32_in1k')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint path or URL. If omitted, randomly initialized '
        'weights are exported (only useful for smoke tests).')
    parser.add_argument(
        '--output',
        default=None,
        help='Output .onnx path. Defaults to <config stem>.onnx in the '
        'current directory.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='Input image shape. Defaults to the shape inferred from the '
        'test pipeline of the config (e.g. 224 224).')
    parser.add_argument(
        '--opset', type=int, default=13, help='ONNX opset version.')
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Export with a dynamic batch dimension.')
    parser.add_argument(
        '--switch-deploy',
        action='store_true',
        help='Call switch_to_deploy() on submodules before export. Required '
        'for re-parameterizable models such as RepVGG / MobileOne.')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify the exported model with onnxsim (must be installed).')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Building model from: {args.config}')
    model = build_model(args.config, args.checkpoint, device='cpu')
    if args.checkpoint is None:
        print('WARNING: no checkpoint given, exporting randomly initialized '
              'weights.')

    if args.switch_deploy:
        num = switch_to_deploy(model)
        print(f'switch_to_deploy() applied to {num} module(s).')

    if args.shape is not None:
        height, width = args.shape
    else:
        height, width = get_input_shape(getattr(model, '_config', None))
        print(f'Input shape inferred from test pipeline: {height}x{width}')

    output = args.output
    if output is None:
        stem = osp.splitext(osp.basename(str(args.config)))[0]
        output = f'{stem}.onnx'

    wrapper = ExportWrapper(model).eval()
    dummy = torch.randn(1, 3, height, width)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {'input': {0: 'batch'}, 'logits': {0: 'batch'}}

    print(f'Exporting to {output} (opset {args.opset}) ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy, ),
            output,
            input_names=['input'],
            output_names=['logits'],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
        )

    import onnx
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    print('onnx.checker passed.')

    if args.simplify:
        try:
            from onnxsim import simplify  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                'onnxsim is not installed, run `pip install onnxsim` or drop '
                'the --simplify flag.')
        simplified, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError('onnxsim failed to validate the simplified '
                               'model.')
        onnx.save(simplified, output)
        print('Model simplified with onnxsim.')

    custom_ops = check_custom_ops(output)
    if custom_ops:
        print('WARNING: ' + format_custom_ops(custom_ops))
    else:
        print('No custom operators found; the model can run on plain '
              'ONNX Runtime.')

    print_onnx_io(output)
    print('Done.')


if __name__ == '__main__':
    main()
