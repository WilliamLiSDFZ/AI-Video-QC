# AI-Video-QC

AI 生成视频质量检测 baseline：给定一段 AI 生成的视频和参照物的实拍照片，
先用 VBench 做本地初筛（可选），再用 ffmpeg 随机抽取若干帧，
调用 Claude API 检测变形、不符合常理、与参照物不一致等问题。

```
VBench 初筛（本地模型，不花 API 费用）
  ├─ 不达标 → 直接出报告，跳过 Claude（exit code 2）
  └─ 达标/未安装 → ffmpeg 抽帧 → Claude 检测 → 汇总报告
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

需要本机已安装 ffmpeg（`brew install ffmpeg`）。

API 密钥放 `.env` 文件（参考 `.env.example`，已被 gitignore）：

```bash
cp .env.example .env   # 然后填入 ANTHROPIC_API_KEY
```

也可直接用同名环境变量；**环境变量优先于 `.env`**。

## 用法

```bash
# 纯命令行传参
.venv/bin/python main.py --video demo.mp4 --ref ref.jpg

# 或把运行参数写进 YAML 配置（模板见 config.example.yaml），CLI 只传素材
cp config.example.yaml config.yaml
.venv/bin/python main.py --video demo.mp4 --ref ref.jpg --config config.yaml
```

`--video` 和 `--ref` 必须由命令行传入；其余参数均可写进配置文件。
优先级：**CLI 显式传参 > `--config` YAML > 内置默认值**。

常用参数：

| 参数 | 说明 | 对应配置字段 |
|---|---|---|
| `--video` | AI 生成的视频文件（必填） | —（仅 CLI） |
| `--ref` | 参照物实拍照片，可传多张（必填） | —（仅 CLI） |
| `--config` | YAML 配置文件（可选） | — |
| `--prompt` | 生成该视频所用的原始 prompt 文本 | `prompt` |
| `--prompt-file` | 从文件读取原始 prompt，与 `--prompt` 二选一 | `prompt_file` |
| `--frames` | 抽帧数量，默认 5 | `frames` |
| `--seed` | 随机种子，固定后可复现抽帧位置 | `seed` |
| `--model` | Claude 模型，默认 `claude-opus-4-8` | `model` |
| `--out-dir` | 输出目录，默认 `output/` | `out_dir` |
| `--no-prescreen` | 跳过 VBench 初筛 | `prescreen.enabled` |
| `--force` | 初筛不通过时仍继续 Claude 检测 | `prescreen.force` |
| `--prescreen-dims` | 初筛维度（逗号分隔），默认全部 6 维 | `prescreen.dims` |
| （仅配置文件） | 初筛阈值覆盖 / 超时 / VBench 解释器路径 | `prescreen.thresholds` / `timeout` / `vbench_python` |

提供原始 prompt 后，检测会额外做两件事：

1. 把 prompt 拆解为逐条要求，逐条判断实现状态（met / partially_met / not_met /
   cannot_judge——静态抽帧无法判断的动作、时序类要求会如实标注，不会乱猜），
   并给出独立的 Prompt 实现度分数（0-10）
2. 结合 prompt 语境判断缺陷：prompt 明确要求的风格化效果不会被误报为缺陷

抽帧方式为分层随机：视频时长均分为 N 段，每段内随机取一个时间点，保证覆盖全片。

## 输出

- 终端打印中文摘要：VBench 初筛逐维分数、画面质量分（0-10）、总体评价、
  与参照物一致性、逐帧问题列表、Prompt 实现度
- `output/<视频名>/report.json`：完整结构化报告（含初筛结果、抽帧时间点、
  模型、seed 等元信息；`verdict` 为 `completed` 或 `prescreen_failed`）
- `output/<视频名>/frames/`：抽出的帧图片，便于人工对照复核

## VBench 初筛

初筛用 [VBench](https://github.com/Vchitect/VBench) 的 `custom_input` 模式在本地评测
6 个客观维度，明显不合格的视频直接拦下，不再花 Claude API 的费用：

| 维度 | 默认阈值 | 说明 |
|---|---|---|
| subject_consistency | ≥ 0.85 | 主体跨帧一致性 |
| background_consistency | ≥ 0.90 | 背景跨帧一致性 |
| motion_smoothness | ≥ 0.95 | 运动平滑度 |
| aesthetic_quality | ≥ 0.45 | 美学质量 |
| imaging_quality | ≥ 0.55 | 成像质量 |
| dynamic_degree | 不参与判定 | 衡量动态程度而非质量，仅报告 |

任一维度低于阈值即判不通过。阈值是经验起点，建议先用已知的好/坏样例各跑一次，
对比分数后写进配置文件的 `prescreen.thresholds` 覆盖。

**未安装 VBench 时初筛自动跳过**（打印提示），不影响主流程。

### 安装 VBench 环境（独立于主 venv）

VBench 依赖较重（PyTorch + 模型权重约 2-4GB）且需要 Python 3.10，
主管线通过子进程调用它，两个环境互不干扰。

**Linux / CUDA 机器（推荐）：**

```bash
conda create -n vbench python=3.10 -y
conda activate vbench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vbench
```

**Apple Silicon Mac（CPU 运行，较慢）：**

```bash
conda create -n vbench python=3.10 -y
conda activate vbench
pip install torch torchvision          # CPU 版
pip install vbench --no-deps           # 官方依赖里的 decord 在 arm64 上装不上
pip install -r requirements-vbench.txt # 手动补齐依赖（decord 已换成 eva-decord）
```

装好后主管线会自动找到 conda env `vbench`；也可用环境变量显式指定解释器：

```bash
export VBENCH_PYTHON=/opt/miniconda3/envs/vbench/bin/python
```

说明：

- 首次运行会自动下载各维度的模型权重到 `~/.cache/vbench`（GB 级）
- 无 GPU 时自动用 CPU，单视频全 6 维在 CPU 上可能需要数分钟到十几分钟
- 个别维度运行失败不影响其余维度出分，失败维度在报告中标注、不参与判定
