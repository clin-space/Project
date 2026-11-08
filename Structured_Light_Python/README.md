# Structured Light Python

这是对相邻 MATLAB 项目中结构光/D2NN 仿真流程的轻量 Python 实现。它保留了核心流程：

1. 生成归一化的 Laguerre-Gaussian (LG) 输入模式。
2. 使用角谱法模拟自由空间传播，并经过透镜傅里叶传播。
3. 以 MATLAB 相同的思路，将奇数边长的复 Fourier 系数网格嵌入相位屏频谱，通过逆 FFT 生成相位，并可选叠加 subharmonics；通过 SPSA 优化复 Fourier 系数。
4. 按目标输出通道计算模式可见度、通道准确率、MATLAB 效率和损耗。
5. 导出训练曲线、输出强度图、指标 CSV、配置 JSON 和系数 NPZ。

默认训练参数按 MATLAB 数值仿真设置：600×600 计算网格、300×300 屏幕、5 层、11×11 Fourier 系数网格、5000 epochs；LG 输入束腰为 0.4 mm，输出通道尺寸为 0.5 mm。SPSA 初始步长/扰动为 5000/0.1，衰减指数为 0.18/0.101；每 10 轮执行 failsafe 检查，cost 改善容差为 2e-4。若缩小计算网格且未指定屏幕大小，屏幕会自动设为 `min(300, grid_size // 2)`。完整默认运行会比较耗时；先用较少 epochs 做试跑。

相位屏只在中央方形区域透光，外围补零；默认蜂窝通道的坐标取整和单轴镜像遵循 MATLAB 仿真。目标高斯光斑按 MATLAB 的有限尺寸采样、补零、最近邻重采样和平移流程生成。默认 0.5 mm 通道宽度在 8 μm 像素间距下对应 62.5 像素，MATLAB 的 `linspace` 实际生成 62 个采样点，因此通道间距和测量半径也采用 62 像素的宽度。`efficiency` 按 MATLAB 的定义计算：每个输出通道汇总所有输入模式的能量，再除以同序号输入模式的总输出能量，最后对通道取平均。通道准确率按每个输出通道归一化；可见度按每个输入模式归一化。cost 使用 MATLAB 的基础权重 `[1, 0, 3, 2, 3]`，其中效率项直接使用原始 `efficiency`，不加入额外的正确通道奖励。

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

九种 LG 模式的拓扑荷数依次为 -4 到 4；输出图中的彩色圆圈标示目标通道，图下方图例给出颜色与荷数的对应关系：

```bash
python -m structured_light --modes 9 --output-dir results/server_run_9
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
- `simulation.png`：输入 LG 模式的相位彩色图（色相表示相位，亮度表示振幅）、目标/输出强度图（均叠加通道圆圈）、相位及通道颜色图例。
- `training_metrics.png`：在一张独立图中叠加 cost、可见度和 MATLAB 式通道效率三条曲线。

已有结果可直接重绘图片，无需重新训练；重绘读取 `metrics.csv`、`summary.json` 和 `phase_coefficients.npz`：

```bash
python -B -c "from structured_light.simulation import redraw_saved_plot; redraw_saved_plot('results/server_run_7')"
```

旧结果若使用了 subharmonics，则因未保存其系数，无法从现有文件恢复输出图像。

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

该实现侧重于可复现的数值仿真，不包含 MATLAB 项目中的 SLM 硬件控制、实验标定、矢量偏振输入或 Skyrmion 测量流程。SPSA 主参数、输入尺寸、传播起点和周期 failsafe 已按 MATLAB 数值仿真设置；训练状态与 MATLAB 的 `coeff_step` 行为仍不同。为避免失败恢复时出现负权重和接近零的权重总和，Python 对动态调整后的 MSE 权重设有零下限。九模式在缩小的网格上可能出现通道重叠，使 MATLAB 式 `efficiency` 大于 1；该原始值仍直接进入 cost，因此 cost 也可能低于零。

## 来源

此项目参考 [Structured_Light_classes](https://github.com/HadrianBez/Structured_Light_classes) 中的 D2NN 仿真、LG 模式、角谱传播和 SPSA 训练思路。原项目采用 MIT License；该 Python 项目为独立实现。
