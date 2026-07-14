# Copyright (c) OpenMMLab. All rights reserved.
"""Verify an exported ONNX model against the original PyTorch model.

Feeds identical inputs to both ``model(x, mode='tensor')`` (PyTorch) and the
ONNX model (ONNX Runtime), then compares the output logits:

- max absolute error and cosine similarity of the logits;
- top-1 / top-5 label agreement after softmax.

Two input modes:
- random tensors (default): several batches generated with a fixed seed;
- a real image (``--img``): preprocessed with the exact test pipeline +
  data_preprocessor of the config, additionally printing the top-5 classes
  predicted by both sides.

Exits with a non-zero code if the max absolute error exceeds ``--atol`` or
the top-1 labels disagree.

Example:
    python verify_onnx.py \
        ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
        resnet18.onnx \
        --checkpoint ../mmpretrain/weights/resnet18_8xb32_in1k.pth \
        --img ../mmpretrain/demo/demo.JPEG
"""
import argparse
import sys

import numpy as np
import torch

from deploy_utils import (build_model, build_preprocess, check_custom_ops,
                          format_custom_ops, switch_to_deploy)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify an ONNX model against the PyTorch model.')
    parser.add_argument(
        'config',
        help='Path to a config file (.py) or a model name in the ModelHub.')
    parser.add_argument('onnx_file', help='Path to the exported .onnx file.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint path or URL. Must match what was used at export '
        'time, otherwise the comparison is meaningless.')
    parser.add_argument(
        '--img',
        default=None,
        help='Optional real image. If given, it is preprocessed with the '
        'test pipeline of the config and the top-5 classes of both sides '
        'are printed.')
    parser.add_argument(
        '--num-batches',
        type=int,
        default=3,
        help='Number of random input batches to compare. Ignored when '
        '--img is given.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size of random inputs. Values > 1 require a model '
        'exported with --dynamic-batch.')
    parser.add_argument(
        '--switch-deploy',
        action='store_true',
        help='Call switch_to_deploy() before comparison. Must match the '
        'export-time setting.')
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-4,
        help='Max absolute error tolerance on logits (fp32).')
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device for the PyTorch model and the preferred ORT execution '
        'provider: "cpu" or "cuda".')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def create_ort_session(onnx_file: str, device: str):
    import onnxruntime as ort

    providers = ['CPUExecutionProvider']
    if device.startswith('cuda'):
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print('WARNING: CUDAExecutionProvider is not available in this '
                  'onnxruntime build, falling back to CPU. Install '
                  'onnxruntime-gpu for GPU inference.')
    session = ort.InferenceSession(onnx_file, providers=providers)
    print(f'ONNX Runtime providers: {session.get_providers()}')
    return session


def compare_logits(torch_logits: np.ndarray, ort_logits: np.ndarray,
                   atol: float, tag: str) -> bool:
    """Compare two logits arrays and print metrics. Returns pass/fail."""
    max_abs_err = float(np.abs(torch_logits - ort_logits).max())

    a = torch_logits.reshape(torch_logits.shape[0], -1).astype(np.float64)
    b = ort_logits.reshape(ort_logits.shape[0], -1).astype(np.float64)
    cos = (a * b).sum(1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    min_cos = float(cos.min())

    def topk(x, k):
        return np.argsort(-x, axis=1)[:, :k]

    k = min(5, torch_logits.shape[1])
    torch_topk, ort_topk = topk(torch_logits, k), topk(ort_logits, k)
    top1_match = bool((torch_topk[:, 0] == ort_topk[:, 0]).all())
    topk_match = bool((torch_topk == ort_topk).all())

    passed = max_abs_err <= atol and top1_match
    status = 'PASS' if passed else 'FAIL'
    print(f'[{tag}] max_abs_err={max_abs_err:.3e}  '
          f'cos_sim={min_cos:.6f}  top1_match={top1_match}  '
          f'top{k}_match={topk_match}  -> {status}')
    return passed


def print_topk(logits: np.ndarray, classes, tag: str, k: int = 5):
    scores = torch.softmax(torch.from_numpy(logits[0]), dim=0).numpy()
    k = min(k, scores.shape[0])
    idx = np.argsort(-scores)[:k]
    print(f'  {tag} top-{k}:')
    for i in idx:
        name = classes[i] if classes is not None else f'class_{i}'
        print(f'    {i:>5d}  {scores[i]:.4f}  {name}')


def main():
    args = parse_args()

    custom_ops = check_custom_ops(args.onnx_file)
    if custom_ops:
        print('ERROR: ' + format_custom_ops(custom_ops))
        sys.exit(2)

    print(f'Building PyTorch model from: {args.config}')
    model = build_model(args.config, args.checkpoint, device=args.device)
    if args.checkpoint is None:
        print('WARNING: no checkpoint given, comparing randomly initialized '
              'weights; make sure the export used the same random state or '
              'results will differ.')
    if args.switch_deploy:
        num = switch_to_deploy(model)
        print(f'switch_to_deploy() applied to {num} module(s).')

    session = create_ort_session(args.onnx_file, args.device)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # e.g. ['batch', 3, 224, 224]
    height, width = int(input_shape[2]), int(input_shape[3])

    def run_both(inputs: np.ndarray):
        with torch.no_grad():
            torch_out = model(
                torch.from_numpy(inputs).to(args.device), mode='tensor')
        torch_logits = torch_out.cpu().numpy()
        # session.run() is typed as a union; for classification logits we
        # always get a dense ndarray as the first (and only) output.
        ort_logits = np.asarray(session.run(None, {input_name: inputs})[0])
        return torch_logits, ort_logits

    all_passed = True

    if args.img is not None:
        preprocess = build_preprocess(model)
        inputs = preprocess(args.img)
        print(f'Image preprocessed to shape {inputs.shape}.')
        torch_logits, ort_logits = run_both(inputs)
        all_passed &= compare_logits(torch_logits, ort_logits, args.atol,
                                     f'image {args.img}')
        classes = getattr(model, '_dataset_meta', {}).get('classes')
        print_topk(torch_logits, classes, 'PyTorch')
        print_topk(ort_logits, classes, 'ONNX Runtime')
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.num_batches):
            # Roughly the scale of normalized image tensors.
            inputs = rng.standard_normal(
                (args.batch_size, 3, height, width)).astype(np.float32)
            torch_logits, ort_logits = run_both(inputs)
            all_passed &= compare_logits(torch_logits, ort_logits, args.atol,
                                         f'random batch {i}')

    if all_passed:
        print('Verification PASSED.')
    else:
        print('Verification FAILED.')
        sys.exit(1)


if __name__ == '__main__':
    main()
