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

推荐用**统一 conda 环境**，一次装齐主项目 + VBench 全部依赖：

```bash
# Linux（含 CUDA 机器）
conda env create -f environment.yml
conda activate ai-video-qc

# macOS Apple Silicon（decord 需特殊处理，用脚本装）
bash scripts/setup_env.sh
conda activate ai-video-qc
```

> ⚠️ **环境必须是 Python 3.10**。VBench 依赖 `transformers==4.33.2`，其旧版
> tokenizers 在 Python 3.12/3.13 下没有预编译轮子，安装会报
> `error: can't find Rust compiler`。environment.yml 已固定 3.10，请勿改。

不需要 VBench 初筛的话，也可以只装主项目（任意 Python ≥3.10）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

另需本机已安装 ffmpeg（macOS: `brew install ffmpeg`；Linux: `apt install ffmpeg`）。

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

### VBench 环境说明

按「安装」小节创建统一环境 `ai-video-qc` 后即可直接使用——初筛检测到
**当前解释器已装 vbench** 时自动启用，无需任何额外配置。

也支持旧的双环境模式（主项目 venv + 独立 VBench 环境），此时用环境变量
指定 VBench 环境的解释器（可写在 `.env` 里）：

```bash
export VBENCH_PYTHON=/path/to/vbench-env/bin/python
```

说明：

- 首次运行会自动下载各维度的模型权重到 `~/.cache/vbench`（GB 级）
- 无 GPU 时自动用 CPU，单视频全 6 维在 CPU 上可能需要数分钟到十几分钟
- 个别维度运行失败不影响其余维度出分，失败维度在报告中标注、不参与判定

### 常见问题

**安装时报 `Building wheel for tokenizers ... error: can't find Rust compiler`**

环境的 Python 版本不是 3.10（多半是 3.12/3.13）。VBench 依赖
`transformers==4.33.2`，其配套的旧版 tokenizers 没有 py3.12+ 的预编译轮子，
pip 转源码编译才需要 Rust——而且旧版对 3.13 装了 Rust 也编不过。修法：

```bash
conda remove -n ai-video-qc --all -y   # 删掉版本不对的环境（名字按实际）
conda env create -f environment.yml    # 用固定 3.10 的规格重建
```

**Linux 上不要套用 macOS 的 `--no-deps` 安装方式**——Linux 的 decord 有
预编译轮子，直接 `pip install vbench`（environment.yml 已是这么做的）。
