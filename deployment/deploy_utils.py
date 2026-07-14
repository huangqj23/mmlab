# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for exporting mmpretrain models to ONNX and verifying
the exported models with ONNX Runtime."""
import os.path as osp
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import torch

# Standard ONNX domains. Any node outside of these domains (e.g. ``mmcv::``,
# ``mmdeploy::``) requires a custom-op plugin at runtime.
STANDARD_ONNX_DOMAINS = ('', 'ai.onnx', 'ai.onnx.ml', 'ai.onnx.preview.training')

CUSTOM_OP_HINT = (
    'The exported ONNX graph contains custom (non-standard) operators, '
    'usually introduced by mmcv ops (e.g. DeformConv2d) or project-specific '
    'CUDA extensions (e.g. DCNv3 of InternImage). Plain ONNX Runtime cannot '
    'execute them. You need the custom-op plugins shipped with MMDeploy '
    '(https://github.com/open-mmlab/mmdeploy) or an inference backend that '
    'implements these operators.')


def build_model(config: str,
                checkpoint: Optional[str] = None,
                device: str = 'cpu') -> torch.nn.Module:
    """Build an mmpretrain model from a config file path or a model-zoo name.

    Args:
        config: Path to a ``.py`` config file, or a model name registered in
            the mmpretrain ModelHub (e.g. ``resnet18_8xb32_in1k``).
        checkpoint: Optional checkpoint path/URL. If ``None``, the model uses
            randomly initialized weights (still useful for smoke tests).
        device: Device to place the model on.

    Returns:
        The model in eval mode, with ``_config`` attached by ``get_model``.
    """
    from mmpretrain.apis import get_model

    pretrained = checkpoint if checkpoint else False
    # get_model() is untyped; it always returns a nn.Module in practice.
    model = cast(torch.nn.Module, get_model(
        config, pretrained=pretrained, device=device))
    model.eval()
    return model


def switch_to_deploy(model: torch.nn.Module) -> int:
    """Recursively call ``switch_to_deploy()`` on all submodules.

    Required for re-parameterizable models such as RepVGG / MobileOne before
    export, so that the multi-branch training structure is fused into the
    efficient inference structure.

    Returns:
        The number of modules that were switched.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy') and callable(
                module.switch_to_deploy):
            module.switch_to_deploy()
            count += 1
    return count


def get_input_shape(config, default: Tuple[int, int] = (224, 224)
                    ) -> Tuple[int, int]:
    """Infer the input (H, W) from the test pipeline of a config.

    Looks for ``CenterCrop`` / ``Resize`` / ``ResizeEdge`` transforms in
    ``cfg.test_dataloader.dataset.pipeline``. Falls back to ``default`` if
    nothing can be inferred.
    """

    def _to_hw(value) -> Optional[Tuple[int, int]]:
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            # mmcv convention is (width, height) for `scale` of Resize,
            # but crop_size of CenterCrop is (width, height) too; for the
            # common square case this does not matter. Return (h, w).
            return (int(value[1]), int(value[0]))
        return None

    try:
        pipeline = config.test_dataloader.dataset.pipeline
    except AttributeError:
        return default

    shape = None
    for transform in pipeline:
        t_type = transform.get('type', '')
        t_type = t_type if isinstance(t_type, str) else t_type.__name__
        if t_type == 'CenterCrop':
            shape = _to_hw(transform.get('crop_size'))
        elif t_type == 'Resize' and shape is None:
            shape = _to_hw(transform.get('scale'))
        elif t_type == 'ResizeEdge' and shape is None:
            scale = transform.get('scale')
            if isinstance(scale, int):
                shape = (scale, scale)
    return shape or default


def build_preprocess(model: torch.nn.Module) -> Callable:
    """Build a preprocess function that maps an image to a normalized tensor.

    The returned callable replicates exactly what
    ``ImageClassificationInferencer`` + ``ClsDataPreprocessor`` do at test
    time: it runs the test pipeline (without ``LoadImageFromFile``) and then
    the model's own ``data_preprocessor`` (BGR->RGB flip and mean/std
    normalization), guaranteeing pixel-exact parity with the PyTorch side.

    Args:
        model: A model built by :func:`build_model` (must carry ``_config``).

    Returns:
        A callable ``preprocess(img) -> np.ndarray`` where ``img`` is an
        image path or a BGR ``np.ndarray``, and the output is a normalized
        float32 array of shape ``(1, 3, H, W)``.
    """
    from mmcv.image import imread
    from mmengine.dataset import Compose, default_collate
    from mmpretrain.datasets import remove_transform
    from mmpretrain.registry import TRANSFORMS

    config = getattr(model, '_config', None)
    if config is None:
        raise ValueError('The model has no attached `_config`. Build it via '
                         'deploy_utils.build_model().')

    pipeline_cfg = remove_transform(
        list(config.test_dataloader.dataset.pipeline), 'LoadImageFromFile')
    transforms = []
    for t in pipeline_cfg:
        transforms.append(TRANSFORMS.build(t) if isinstance(t, dict) else t)
    pipeline = Compose(transforms)
    # Module attribute access is loosely typed in torch stubs.
    data_preprocessor = cast(Callable[..., Any], model.data_preprocessor)

    def preprocess(img: Union[str, np.ndarray]) -> np.ndarray:
        img = imread(img)  # BGR ndarray
        if img is None:
            raise ValueError('Failed to read the input image.')
        data = pipeline(
            dict(img=img, img_shape=img.shape[:2], ori_shape=img.shape[:2]))
        batch = default_collate([data])
        with torch.no_grad():
            inputs = data_preprocessor(batch, False)['inputs']
        return inputs.cpu().numpy().astype(np.float32)

    return preprocess


def check_custom_ops(onnx_path: str) -> List[Tuple[str, str]]:
    """Scan an ONNX file for nodes from non-standard domains.

    Returns:
        A list of unique ``(domain, op_type)`` pairs of custom operators.
        Empty list means the model only uses standard ONNX operators and can
        run on plain ONNX Runtime.
    """
    import onnx

    graph = onnx.load(onnx_path).graph
    custom = []
    seen = set()
    for node in graph.node:
        if node.domain not in STANDARD_ONNX_DOMAINS:
            key = (node.domain, node.op_type)
            if key not in seen:
                seen.add(key)
                custom.append(key)
    return custom


def format_custom_ops(custom_ops: List[Tuple[str, str]]) -> str:
    """Format the result of :func:`check_custom_ops` for user-facing logs."""
    ops = ', '.join(f'{domain}::{op}' for domain, op in custom_ops)
    return f'Custom operators found: {ops}\n{CUSTOM_OP_HINT}'


def print_onnx_io(onnx_path: str) -> None:
    """Print the inputs/outputs (name, dtype, shape) of an ONNX model."""
    import onnx

    model = onnx.load(onnx_path)

    def _fmt(value_info):
        ttype = value_info.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(ttype.elem_type).lower()
        dims = []
        for d in ttype.shape.dim:
            dims.append(d.dim_param if d.dim_param else str(d.dim_value))
        return f"{value_info.name}: {dtype}[{', '.join(dims)}]"

    initializers = {init.name for init in model.graph.initializer}
    print(f'ONNX model: {osp.abspath(onnx_path)}')
    print(f'  opset: {model.opset_import[0].version}')
    for inp in model.graph.input:
        if inp.name not in initializers:
            print(f'  input  -> {_fmt(inp)}')
    for out in model.graph.output:
        print(f'  output -> {_fmt(out)}')
