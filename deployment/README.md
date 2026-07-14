# mmpretrain ONNX 部署工具

将本仓库 `mmpretrain` 训练的分类模型导出为 ONNX 格式，并用 ONNX Runtime 对导出模型做精度验证。本目录独立于各子模块（mmpretrain / mmcv / mmengine），不修改子模块内容。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `export_onnx.py` | 将 config + checkpoint 导出为 ONNX（含导出后 `onnx.checker` 校验和自定义算子检测） |
| `verify_onnx.py` | 用相同输入对比 PyTorch 与 ONNX Runtime 的输出（误差、余弦相似度、top-1/top-5 一致性） |
| `deploy_utils.py` | 公共逻辑：建模、`switch_to_deploy`、前处理复刻、自定义算子检测 |
| `requirements.txt` | 部署侧额外依赖 |

## 安装依赖

在已装好 mmpretrain/mmcv/mmengine/torch 的环境（如 conda 环境 `mmlab`）中：

```bash
pip install -r requirements.txt
```

## 导出 ONNX

```bash
cd deployment

# 使用 config 文件 + checkpoint
python export_onnx.py \
    ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
    --checkpoint ../mmpretrain/weights/resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
    --output resnet18.onnx \
    --dynamic-batch

# 也可直接使用 model zoo 名称（联网自动取配置）
python export_onnx.py resnet18_8xb32_in1k --checkpoint <ckpt> --output resnet18.onnx
```

常用参数：

- `--shape H W`：输入尺寸，默认从配置的 test_pipeline 自动推断（如 CenterCrop 224）。
- `--opset`：ONNX opset 版本，默认 13。
- `--dynamic-batch`：batch 维导出为动态。
- `--switch-deploy`：导出前对子模块调用 `switch_to_deploy()`，RepVGG / MobileOne 等重参数化模型必须加。
- `--simplify`：用 onnxsim 化简图（需 `pip install onnxsim`）。

### 导出约定

导出的图输入是**归一化后的** float32 张量 `(N, 3, H, W)`（即已完成 resize/crop、BGR→RGB、`(x-mean)/std`），输出是**未过 softmax 的 logits** `(N, num_classes)`。前处理和 softmax 留在图外，由部署侧自行实现（`deploy_utils.build_preprocess` 提供了与训练侧逐像素一致的参考实现）。

## 验证 ONNX

```bash
# 随机张量对比（默认 3 组，固定 seed）
python verify_onnx.py \
    ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
    resnet18.onnx \
    --checkpoint ../mmpretrain/weights/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

# 真实图片对比（额外打印双方 top-5 类别）
python verify_onnx.py \
    ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
    resnet18.onnx \
    --checkpoint <ckpt> \
    --img ../mmpretrain/demo/demo.JPEG
```

判定标准：logits 最大绝对误差不超过 `--atol`（默认 `1e-4`，fp32）且 top-1 标签一致，否则退出码非零。`--device cuda` 可让 PyTorch 侧走 GPU（ORT 侧需要安装 `onnxruntime-gpu` 才会启用 CUDAExecutionProvider）。

注意：`--checkpoint` 与 `--switch-deploy` 必须与导出时保持一致，否则对比无意义。

## 关于 mmcv 自定义算子

mmpretrain 核心分类模型（ResNet / ResNeXt / ViT / Swin / MobileNet 等）只使用 `mmcv.cnn` 中的标准模块，导出的 ONNX 全部为标准算子，可直接在原生 ONNX Runtime 上运行。

但部分模型会引入非标准算子（ONNX 图中出现 `mmcv::` 或 `mmdeploy::` 域的节点），例如：

- 使用 `mmcv.ops` 自定义算子（DeformConv2d、ModulatedDeformConv2d 等）的自定义 backbone；
- `mmpretrain/projects/internimage_classification`（DCNv3 自定义 CUDA 扩展）。

两个脚本都会自动扫描导出图中的非标准域算子：`export_onnx.py` 发现时打印警告，`verify_onnx.py` 发现时直接报错退出。这类模型无法在原生 ONNX Runtime 上运行，需要 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 提供的自定义算子插件（mmcv 2.x 本身不再附带 ONNX Runtime 插件），不在本工具支持范围内。

## 重参数化模型（RepVGG / MobileOne 等）

这类模型训练结构与部署结构不同，导出与验证时都要加 `--switch-deploy`：

```bash
python export_onnx.py ../mmpretrain/configs/repvgg/repvgg-A0_8xb32_in1k.py \
    --checkpoint <ckpt> --switch-deploy --output repvgg-a0.onnx
python verify_onnx.py ../mmpretrain/configs/repvgg/repvgg-A0_8xb32_in1k.py \
    repvgg-a0.onnx --checkpoint <ckpt> --switch-deploy
```

若 checkpoint 本身已是 deploy 结构（例如经 `tools/model_converters/reparameterize_model.py` 转换），则配置中 `backbone.deploy=True` 即可，无需该参数。
