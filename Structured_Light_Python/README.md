# Structured Light Python

这是对相邻 MATLAB 项目中结构光/D2NN 仿真流程的轻量 Python 实现。它保留了核心流程：

1. 生成归一化的 Laguerre-Gaussian (LG) 输入模式。
2. 使用角谱法模拟自由空间传播，并经过透镜傅里叶传播。
3. 以 MATLAB 相同的思路，将奇数边长的复 Fourier 系数网格嵌入相位屏频谱，通过逆 FFT 生成相位，并可选叠加 subharmonics；通过 SPSA 优化复 Fourier 系数。
4. 按目标输出通道计算模式可见度、通道准确率、效率和损耗。
5. 导出训练曲线、输出强度图、指标 CSV、配置 JSON 和系数 NPZ。

默认训练参数按 MATLAB 数值仿真设置：600×600 计算网格、300×300 屏幕、5 层、11×11 Fourier 系数网格、5000 epochs；LG 输入束腰为 0.4 mm，输出通道尺寸为 0.5 mm。SPSA 初始步长/扰动为 5000/0.1，衰减指数为 0.18/0.101；每 10 轮执行 failsafe 检查，cost 改善容差为 2e-4。若缩小计算网格且未指定屏幕大小，屏幕会自动设为 `min(300, grid_size // 2)`。完整默认运行会比较耗时；先用较少 epochs 做试跑。

## 环境安装

需要 Python 3.10 或更新版本。

```bash
cd Structured_Light_Python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

默认使用 PyTorch 后端。安装可选 Torch 依赖：

```bash
python -m pip install -e ".[torch]"
```

当前环境中 PyTorch 的 `cuda:0` 是 RTX 4090，因此默认使用 `cuda:0`。也可通过参数指定其他支持的设备：

```bash
python -m structured_light --epochs 100
```

可显式使用 `--device cuda:1`、`--device cuda`、`--device auto` 或 `--device cpu`。若指定的 CUDA 设备不可见，程序会报告错误。未安装 PyTorch 时会显示安装提示。旧的 NumPy 后端仍可通过 `--backend numpy` 选择。

## 运行

```bash
python -m structured_light --epochs 5000
```

指定网格、随机种子和输出目录：

```bash
python -m structured_light --grid-size 600 --phase-screen-size 300 --screens 5 --epochs 100 --seed 7 --output-dir results
```

显式设定相位屏和低频 subharmonics：

```bash
python -m structured_light --grid-size 600 --phase-screen-size 300 --fourier-basis-size 11 --num-subharmonics 2 --epochs 100
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

可通过 `SimulationConfig` 调整网格和相位屏尺寸、Fourier 系数网格、subharmonics、模式/屏幕数量、传播参数、训练轮数、SPSA 步长/扰动及其衰减指数、failsafe 检查设置。Python 调用示例：

```python
from structured_light import SimulationConfig, run_simulation

result = run_simulation(SimulationConfig(grid_size=96, phase_screen_size=48, epochs=5))
print(result.final_metrics)
```

`SimulationConfig` 默认使用 Torch 和 `device="cuda:0"`；需要兼容旧计算路径时可传入 `backend="numpy"`。

该实现侧重于可复现的数值仿真，不包含 MATLAB 项目中的 SLM 硬件控制、实验标定、矢量偏振输入或 Skyrmion 测量流程。SPSA 主参数、输入尺寸、传播起点和周期 failsafe 已按 MATLAB 数值仿真设置；仍与 MATLAB 不同的是额外的实验/硬件流程及部分失败时的 cost 权重保护细节。初始 cost 权重为 `[1, 0, 3, 2, 3]`。

## 来源

此项目参考 [Structured_Light_classes](https://github.com/HadrianBez/Structured_Light_classes) 中的 D2NN 仿真、LG 模式、角谱传播和 SPSA 训练思路。原项目采用 MIT License；该 Python 项目为独立实现。
