# Structured Light Python

这是对相邻 MATLAB 项目中结构光/D2NN 仿真流程的轻量 Python 实现。它保留了核心流程：

1. 生成归一化的 Laguerre-Gaussian (LG) 输入模式。
2. 使用角谱法模拟自由空间传播，并经过透镜傅里叶传播。
3. 以 MATLAB 相同的思路，将奇数边长的复 Fourier 系数网格嵌入相位屏频谱，通过逆 FFT 生成相位，并可选叠加 subharmonics；通过 SPSA 优化复 Fourier 系数。
4. 按目标输出通道计算模式可见度、通道准确率、效率和损耗。
5. 导出训练曲线、输出强度图、指标 CSV、配置 JSON 和系数 NPZ。

默认参数是为首次运行和普通电脑设计的快速示例：96×96 计算网格、48×48 屏幕、11×11 Fourier 系数网格和 10 个 epochs，不等同于原 MATLAB 文件中的 600×600、300×300 屏幕和 5000 epochs 配置。屏幕默认取计算网格边长的一半。可通过命令行增加网格大小和训练轮数。

## 环境安装

需要 Python 3.10 或更新版本。

```bash
cd Structured_Light_Python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## 运行

```bash
python -m structured_light --epochs 10
```

指定网格、随机种子和输出目录：

```bash
python -m structured_light --grid-size 128 --epochs 50 --seed 7 --output-dir results
```

显式设定相位屏和低频 subharmonics：

```bash
python -m structured_light --grid-size 128 --phase-screen-size 64 --fourier-basis-size 11 --num-subharmonics 2
```

程序会在输出目录中生成：

- `metrics.csv`：每轮 SPSA 测量结果。
- `summary.json`：最终指标及配置。
- `phase_coefficients.npz`：优化得到的相位屏 Fourier 系数。
- `simulation.png`：输入/目标/输出强度与训练曲线。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 参数与 Python API

可通过 `SimulationConfig` 调整 `grid_size`、相位屏大小、Fourier 系数网格尺寸、subharmonics 层数、输入模式数、相位屏层数、波长、像素大小、传播距离、输出通道尺寸、训练轮数和随机种子。Python 调用示例：

```python
from structured_light import SimulationConfig, run_simulation

result = run_simulation(SimulationConfig(grid_size=96, epochs=5))
print(result.final_metrics)
```

该实现侧重于可复现的数值仿真与可运行示例，不包含 MATLAB 项目中的 SLM 硬件控制、实验标定、矢量偏振输入或 Skyrmion 测量流程。

## 来源

此项目参考 [Structured_Light_classes](https://github.com/HadrianBez/Structured_Light_classes) 中的 D2NN 仿真、LG 模式、角谱传播和 SPSA 训练思路。原项目采用 MIT License；该 Python 项目为独立实现。
