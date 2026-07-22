"""VBench 初筛：在独立环境中跑 VBench custom_input 评测，按阈值判定视频是否值得送 Claude 检测。

VBench 依赖较重且需要独立的 Python 3.10 环境（见 README「VBench 初筛」小节），
因此这里通过 subprocess 调用 scripts/run_vbench.py，主环境不引入任何 VBench 依赖。
未安装 VBench 环境时初筛自动跳过，不阻塞主流程。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ALL_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# 经验起点阈值，建议用已知的好/坏样例实测标定后通过 --prescreen-config 覆盖。
# dynamic_degree 衡量动态程度而非质量（静态产品视频天然低分），不参与判定，仅报告。
DEFAULT_THRESHOLDS = {
    "subject_consistency": 0.85,
    "background_consistency": 0.90,
    "motion_smoothness": 0.95,
    "aesthetic_quality": 0.45,
    "imaging_quality": 0.55,
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INSTALL_HINT = ("未找到 VBench 环境，初筛已跳过。安装方法见 README「VBench 初筛」小节，"
                "或用 VBENCH_PYTHON 环境变量指定解释器路径")


def _runner_path() -> Path:
    # VBENCH_RUNNER 仅用于测试替换 runner 脚本
    override = os.environ.get("VBENCH_RUNNER")
    if override:
        return Path(override)
    return _PROJECT_ROOT / "scripts" / "run_vbench.py"


@dataclass
class PrescreenResult:
    status: str  # "passed" | "failed" | "skipped"
    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    failed_dims: list[str] = field(default_factory=list)  # 低于阈值的维度
    thresholds: dict[str, float] = field(default_factory=dict)
    device: str | None = None
    note: str = ""


def find_vbench_python() -> Path | None:
    # 1. 显式指定优先
    env_python = os.environ.get("VBENCH_PYTHON")
    if env_python and Path(env_python).is_file():
        return Path(env_python)
    # 2. 统一环境模式：当前解释器已装 vbench，直接用（零配置）
    if importlib.util.find_spec("vbench") is not None:
        return Path(sys.executable)
    # 3. 旧的双环境模式
    candidates = [
        _PROJECT_ROOT / ".venv-vbench" / "bin" / "python",
        Path("/opt/miniconda3/envs/vbench/bin/python"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_prescreen(video: Path, dims: list[str] | None = None,
                  thresholds: dict[str, float] | None = None,
                  timeout: int = 3600,
                  python: Path | None = None) -> PrescreenResult:
    dims = dims or list(ALL_DIMS)
    merged_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if python is not None:
        if not Path(python).is_file():
            return PrescreenResult(
                status="skipped", note=f"指定的 VBench 解释器不存在: {python}")
    else:
        python = find_vbench_python()
    if python is None:
        return PrescreenResult(status="skipped", note=INSTALL_HINT)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "vbench_result.json"
        try:
            proc = subprocess.run(
                [str(python), str(_runner_path()), "--video", str(video),
                 "--dims", ",".join(dims), "--out", str(out)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return PrescreenResult(
                status="skipped", note=f"VBench 运行超时（>{timeout}s），初筛跳过")
        if not out.is_file():
            detail = (proc.stderr.strip() or proc.stdout.strip())[-500:]
            return PrescreenResult(
                status="skipped", note=f"VBench 运行失败，初筛跳过: {detail}")
        data = json.loads(out.read_text(encoding="utf-8"))

    scores = {k: float(v) for k, v in data.get("scores", {}).items()}
    errors = data.get("errors", {})
    device = data.get("device")
    if not scores:
        return PrescreenResult(status="skipped", errors=errors, device=device,
                               note="所有维度均运行失败，初筛跳过")

    # 任一参与判定的维度低于阈值 → 不通过；运行失败的维度视为未知，不判 fail
    failed = [d for d, s in scores.items()
              if d in merged_thresholds and s < merged_thresholds[d]]
    return PrescreenResult(
        status="failed" if failed else "passed",
        scores=scores,
        errors=errors,
        failed_dims=failed,
        thresholds={d: merged_thresholds[d] for d in scores if d in merged_thresholds},
        device=device,
    )
