#!/usr/bin/env bash
# 一键创建统一 conda 环境 ai-video-qc（主项目 + VBench 全部依赖）。
#
# 用法:
#   bash scripts/setup_env.sh            # 实际安装
#   bash scripts/setup_env.sh --dry-run  # 只打印将执行的命令
#
# Linux 直接套用 environment.yml；macOS Apple Silicon 走 eva-decord 兼容路径。
set -euo pipefail

ENV_NAME=ai-video-qc
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

if ! command -v conda >/dev/null 2>&1; then
    echo "错误: 未找到 conda，请先安装 miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

# 已存在的环境必须是 Python 3.10.x，否则 VBench 的旧版 tokenizers 装不上
if [[ -x "$ENV_PYTHON" ]]; then
    PYVER="$("$ENV_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    if [[ "$PYVER" != "3.10" ]]; then
        echo "错误: conda 环境 $ENV_NAME 已存在但 Python 版本是 $PYVER（需要 3.10）。" >&2
        echo "VBench 依赖 transformers==4.33.2，其旧版 tokenizers 在 py3.12+ 无预编译轮子。" >&2
        echo "请先删除后重跑本脚本: conda remove -n $ENV_NAME --all -y" >&2
        exit 1
    fi
    echo "环境 $ENV_NAME 已存在（Python $PYVER），跳过创建，仅补装依赖。"
    ENV_EXISTS=1
else
    ENV_EXISTS=0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    # Apple Silicon：decord 无 arm64 轮子，vbench 用 --no-deps 装，
    # 依赖由 requirements-vbench.txt（decord 已换成 eva-decord）补齐
    if [[ "$ENV_EXISTS" -eq 0 ]]; then
        run conda create -n "$ENV_NAME" python=3.10 -y
    fi
    run "$ENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"
    run "$ENV_PYTHON" -m pip install torch torchvision
    run "$ENV_PYTHON" -m pip install vbench --no-deps
    run "$ENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements-vbench.txt"
else
    # Linux（含 CUDA 机器）：PyPI 的 torch 轮子自带 CUDA runtime，直接装
    if [[ "$ENV_EXISTS" -eq 0 ]]; then
        run conda env create -f "$PROJECT_ROOT/environment.yml"
    else
        run conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
    fi
fi

echo
echo "校验安装 ..."
run "$ENV_PYTHON" -c "import vbench, torch, anthropic; print('vbench / torch / anthropic 导入成功, cuda:', torch.cuda.is_available())"

echo
echo "完成。使用方法:"
echo "  conda activate $ENV_NAME"
echo "  python main.py --video demo.mp4 --ref ref.jpg"
echo "（统一环境下无需配置 VBENCH_PYTHON，初筛自动使用当前解释器）"
