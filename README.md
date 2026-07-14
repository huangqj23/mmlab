# mmlab

基于 OpenMMLab 生态的计算机视觉算法库集合，统一管理训练引擎、基础算子与下游视觉任务算法，便于在同一环境中完成模型训练、评估与部署相关开发。

## 定位

本仓库将 OpenMMLab 系列核心组件与任务算法集中组织，形成可扩展的多任务视觉算法底座：

- **底层支撑**：统一训练流程、配置系统、数据管线与常用视觉算子
- **任务算法**：覆盖分类 / 预训练，并逐步扩展到检测、分割、位姿、3D 与多模态等方向
- **工程目标**：降低多任务切换成本，便于复现实验、二次开发与算法集成

## 当前内容

| 子库 | 说明 |
|------|------|
| [mmengine](./mmengine) | OpenMMLab 下一代训练引擎，提供 Runner、Hook、优化器、日志与分布式训练等基础能力 |
| [mmcv](./mmcv) | 计算机视觉基础库，提供数据变换、可视化、CNN 组件与 CUDA 算子等 |
| [mmpretrain](./mmpretrain) | 图像分类与自监督预训练算法库，支持丰富的骨干网络与预训练方法 |

## 规划中

后续将陆续接入更多 OpenMMLab 任务算法：

- **目标检测**（如 MMDetection）
- **语义 / 实例分割**（如 MMSegmentation）
- **人体 / 物体位姿估计**（如 MMPose）
- **3D 目标检测**（如 MMDetection3D）
- **多模态理解与生成**（如相关多模态算法库）

## 目录结构

```text
mmlab/
├── mmengine/      # 训练引擎
├── mmcv/          # 视觉基础库
├── mmpretrain/    # 分类与预训练
└── README.md
```

## 使用说明

各子库可独立安装与使用，建议按依赖顺序安装：

1. `mmengine`
2. `mmcv`
3. `mmpretrain`（及其他后续任务库）

具体安装方式、配置与训练命令请参考各子库目录下的 `README.md`。

## 参考

- [OpenMMLab](https://openmmlab.com/)
- [MMEngine](https://github.com/open-mmlab/mmengine)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain)
