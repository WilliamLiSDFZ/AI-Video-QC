# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI 生成视频质量检测：给定 AI 生成的视频 + 参照物实拍照片（可附原始生成 prompt），
输出结构化质量报告。所有用户可见输出（CLI 摘要、报错、system prompt、报告内容）均为中文，保持这一约定。

## 常用命令

```bash
# 环境（统一 conda 环境，含 VBench；Linux 一条命令）
conda env create -f environment.yml && conda activate ai-video-qc
# Apple Silicon 用脚本（处理 decord 无 arm64 轮子的问题）
bash scripts/setup_env.sh          # 支持 --dry-run

# 仅主项目（不需要 VBench 初筛时）
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 运行（--video/--ref 必须走 CLI，其余参数可放 YAML）
python main.py --video demo.mp4 --ref ref.jpg [--config config.yaml] [--prompt "..."] [--seed 42]
```

依赖 ffmpeg/ffprobe 在 PATH 中（用 VBench 初筛还需要 `unzip`，见下）。API 密钥放项目根目录 `.env`（`ANTHROPIC_API_KEY`；已 gitignore，模板见 `.env.example`）；已导出的环境变量优先于 `.env`。

没有正式测试套件。验证靠临时脚本 + 下述测试钩子；改动后至少要跑一遍带假 runner 的初筛路径和 mock 掉 API 的消息构造检查。

## 架构

`main.py` 串起三阶段流水线，每阶段可独立失败/跳过：

1. **VBench 初筛**（`video_qc/prescreener.py`）— 本地客观指标，拦下明显差的视频以省 API 费用。不通过 → 保存报告后 **exit code 2**（`--force` 可继续）；VBench 未安装 → 自动跳过不阻塞。
2. **抽帧**（`video_qc/frame_extractor.py`）— 分层随机：时长均分 N 段、每段随机取一点（`--seed` 可复现），首尾留 0.2s 余量。
3. **Claude 检测**（`video_qc/claude_checker.py`）— 单条多图 user 消息（参照图在前、抽帧按时间序带时间戳标注），`client.messages.parse()` + Pydantic `QCReport` 结构化输出。

### 关键设计约束

- **VBench 跨环境子进程隔离**：VBench 硬依赖 `transformers==4.33.2`（旧版 tokenizers 在 py3.12+ 无轮子，装不上，报 "can't find Rust compiler"——环境必须 Python 3.10）。因此 prescreener **绝不在主进程 import vbench**，而是子进程调用 `scripts/run_vbench.py`。该脚本必须保持不依赖主项目代码（它在 VBench 环境里跑）。解释器探测顺序（`find_vbench_python`）：`VBENCH_PYTHON` 环境变量 > 当前解释器已可 import vbench（统一环境模式）> `.venv-vbench` > conda env `vbench`。
- **配置优先级**：CLI 显式传参 > `--config` YAML > 内置默认值，合并逻辑集中在 `video_qc/config.py::load_settings`（argparse 所有可选参数默认 None 以区分"未传"）。YAML 带严格 schema 校验（未知字段报错）。`.env` 只放密钥类，不放运行参数。
- **初筛判定**：`DEFAULT_THRESHOLDS` 逐维阈值，任一低于阈值即不通过；`dynamic_degree` 默认只报告不判定（衡量动态程度而非质量），用户在 YAML `prescreen.thresholds` 里给它设阈值才参与。runner 逐维度 try/except，个别维度失败不影响其他维度出分，失败维度视为未知、不判 fail。
- **两个独立分数**：`overall_score` 只评画面质量；`prompt_adherence_score` 只评 prompt 实现度。system prompt 要求静态抽帧无法判断的 prompt 要求（动作、时序、音频）必须标 `cannot_judge`，禁止模型猜测。
- **报告**：`output/<视频名>/report.json`，`verdict` 为 `completed` 或 `prescreen_failed`；抽帧图片保留在 `output/<视频名>/frames/` 供人工复核。

### 测试钩子

- `VBENCH_RUNNER` 环境变量可替换 runner 脚本路径（仅测试用）：配合 `VBENCH_PYTHON` 指向主环境 python + 一个输出伪造 JSON 的假 runner，即可离线覆盖初筛的通过/不通过/维度失败/崩溃路径。
- Claude 调用测试：`unittest.mock.patch("anthropic.Anthropic")` 截获 `messages.parse` 的 kwargs 检查消息结构，不发真实请求。

### VBench 事实（改初筛相关代码前先知道）

- `custom_input` 模式支持且仅支持 6 维：subject_consistency、background_consistency、motion_smoothness、dynamic_degree、aesthetic_quality、imaging_quality（`prescreener.ALL_DIMS`）。detectron2 只有其他维度需要，不用装。
- 模型权重首次运行自动下载到 `~/.cache/vbench`（GB 级）；runner 内 device 自动选 cuda/cpu。
- VBench 的维度分发是 try/except import，**任何 import 失败都被吞成
  `NotImplementedError: UnImplemented dimension xxx`**，真实原因在附带的消息里。已知三个环境坑：
  ① `No module named 'pkg_resources'` = setuptools 太新（openai-clip/pyiqa 还在用它），
  environment.yml 已固定 `setuptools<81`，影响全部 CLIP 系维度；
  ② dynamic_degree 需要系统 `unzip` 二进制解压 RAFT 权重；
  ③ torch≥2.6 的 `torch.load(weights_only=True)` 默认值加载不了 VBench 旧权重，
  `run_vbench.py` 里已 monkeypatch 回 `False`（权重来源可信）——这三处别在"重构"时删掉。
- `requirements-vbench.txt` 是给 Apple Silicon `pip install vbench --no-deps` 后补依赖用的（decord 换成 eva-decord），Linux 不要用这条路径。

### 待办状态

- 初筛默认阈值是经验起点，尚未用真实样例标定——仓库里的 `good_example.mp4` / `bad_example.mp4` + `adam_bar_916.png` 就是为此准备的标定素材。
