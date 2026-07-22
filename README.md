# AI-Video-QC

AI 生成视频质量检测 baseline：给定一段 AI 生成的视频和参照物的实拍照片，
用 ffmpeg 随机抽取若干帧，调用 Claude API 检测变形、不符合常理、与参照物不一致等问题。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

需要本机已安装 ffmpeg（`brew install ffmpeg`），并配置 Claude API 凭证
（`ANTHROPIC_API_KEY`，或 `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`）。

## 用法

```bash
.venv/bin/python main.py --video demo.mp4 --ref ref.jpg
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--video` | AI 生成的视频文件（必填） |
| `--ref` | 参照物实拍照片，可传多张（必填） |
| `--frames` | 抽帧数量，默认 5 |
| `--seed` | 随机种子，固定后可复现抽帧位置 |
| `--model` | Claude 模型，默认 `claude-opus-4-8` |
| `--out-dir` | 输出目录，默认 `output/` |

抽帧方式为分层随机：视频时长均分为 N 段，每段内随机取一个时间点，保证覆盖全片。

## 输出

- 终端打印中文摘要：总体质量分（0-10）、总体评价、与参照物一致性、逐帧问题列表
- `output/<视频名>/report.json`：完整结构化报告（含抽帧时间点、模型、seed 等元信息）
- `output/<视频名>/frames/`：抽出的帧图片，便于人工对照复核
