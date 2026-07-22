#!/usr/bin/env python3
"""AI 生成视频质量检测 baseline。

流程：VBench 初筛（可选）→ ffmpeg 随机抽帧 → Claude API 检测。
给定一段 AI 生成的视频和参照物实拍照片（可附生成 prompt），
检测变形、不合常理、与参照物或 prompt 不符之处。

密钥放 .env（或环境变量）；运行参数可放 YAML 配置文件（--config，可选），
也可直接用命令行传参。优先级：CLI 显式传参 > --config YAML > 内置默认值。

用法:
    python main.py --video demo.mp4 --ref ref.jpg [--config config.yaml] [--prompt "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from video_qc.claude_checker import QCReport, run_qc
from video_qc.config import ConfigError, Settings, load_settings
from video_qc.frame_extractor import Frame, FrameExtractionError, extract_frames
from video_qc.prescreener import PrescreenResult, run_prescreen

PROJECT_ROOT = Path(__file__).resolve().parent
SEVERITY_MARK = {"low": "·", "medium": "⚠", "high": "✖"}
STATUS_MARK = {"met": "✓", "partially_met": "◐", "not_met": "✗", "cannot_judge": "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI 生成视频质量检测 baseline",
        epilog="除 --video/--ref 外的参数均可写进 YAML 配置文件（--config）；"
               "CLI 显式传参优先于配置文件。")
    parser.add_argument("--video", required=True, type=Path, help="AI 生成的视频文件")
    parser.add_argument("--ref", required=True, type=Path, nargs="+",
                        help="参照物实拍照片（可多张）")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML 配置文件（可选，模板见 config.example.yaml）")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None,
                              help="生成该视频所用的原始 prompt 文本")
    prompt_group.add_argument("--prompt-file", type=Path, default=None,
                              help="从文件读取原始 prompt（utf-8）")
    parser.add_argument("--frames", type=int, default=None, help="抽帧数量（默认 5）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，用于复现抽帧位置")
    parser.add_argument("--model", default=None,
                        help="Claude 模型（默认 claude-opus-4-8）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="输出目录（默认 output/）")
    parser.add_argument("--no-prescreen", action="store_true",
                        help="跳过 VBench 初筛")
    parser.add_argument("--force", action="store_true",
                        help="初筛不通过时仍继续 Claude 检测")
    parser.add_argument("--prescreen-dims", default=None,
                        help="初筛维度，逗号分隔（默认全部 6 维）")
    return parser.parse_args()


def check_env(args: argparse.Namespace, settings: Settings) -> None:
    errors = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        errors.append("未找到 ffmpeg/ffprobe，请先安装: brew install ffmpeg")
    if not args.video.is_file():
        errors.append(f"视频文件不存在: {args.video}")
    for ref in args.ref:
        if not ref.is_file():
            errors.append(f"参照图片不存在: {ref}")
    if settings.prompt_file is not None and not settings.prompt_file.is_file():
        errors.append(f"prompt 文件不存在: {settings.prompt_file}")
    has_cred = (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())
    if not has_cred:
        errors.append("未找到 Claude API 凭证：请在 .env 中填入 ANTHROPIC_API_KEY"
                      "（参考 .env.example），或设置同名环境变量")
    if errors:
        for e in errors:
            print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def print_prescreen(p: PrescreenResult) -> None:
    print("\n【VBench 初筛】")
    if p.status == "skipped":
        print(f"  跳过: {p.note}")
        return
    for dim, score in p.scores.items():
        th = p.thresholds.get(dim)
        if th is None:
            print(f"  - {dim}: {score:.4f}（不参与判定）")
        else:
            mark = "✓" if score >= th else "✗"
            print(f"  {mark} {dim}: {score:.4f}（阈值 {th}）")
    for dim, err in p.errors.items():
        print(f"  ? {dim}: 运行失败（{err}）")
    print(f"  初筛结果: {'通过' if p.status == 'passed' else '不通过'}")


def print_summary(report: QCReport, frames: list[Frame]) -> None:
    ts = {f.index: f.timestamp for f in frames}
    print("\n" + "=" * 60)
    line = (f"画面质量分: {report.overall_score}/10"
            f"    发现缺陷: {'是' if report.has_defects else '否'}")
    if report.prompt_adherence_score is not None:
        line += f"    Prompt 实现度: {report.prompt_adherence_score}/10"
    print(line)
    print("=" * 60)
    print(f"\n【总体评价】\n{report.overall_assessment}")
    print(f"\n【与参照物一致性】\n{report.reference_consistency}")
    print("\n【逐帧检测】")
    for finding in report.frame_findings:
        t = ts.get(finding.frame_index)
        header = f"帧 #{finding.frame_index}" + (f"（t={t:.1f}s）" if t is not None else "")
        if not finding.issues:
            print(f"  {header}: 未发现问题")
            continue
        print(f"  {header}:")
        for issue in finding.issues:
            mark = SEVERITY_MARK.get(issue.severity, "·")
            print(f"    {mark} [{issue.category}/{issue.severity}] {issue.description}")
    if report.prompt_requirements:
        print("\n【Prompt 实现度】")
        if report.prompt_adherence:
            print(f"  {report.prompt_adherence}")
        for check in report.prompt_requirements:
            mark = STATUS_MARK.get(check.status, "?")
            print(f"  {mark} [{check.status}] {check.requirement}")
            print(f"      {check.note}")
    print()


def build_report_data(args: argparse.Namespace, settings: Settings,
                      prompt: str | None,
                      prescreen: PrescreenResult | None,
                      frames: list[Frame] | None = None,
                      report: QCReport | None = None,
                      verdict: str = "completed") -> dict:
    return {
        "video": str(args.video),
        "references": [str(r) for r in args.ref],
        "generation_prompt": prompt,
        "model": settings.model,
        "seed": settings.seed,
        "config_file": str(settings.config_path) if settings.config_path else None,
        "verdict": verdict,
        "prescreen": asdict(prescreen) if prescreen else None,
        "frames": ([{"index": f.index, "timestamp": f.timestamp, "path": str(f.path)}
                    for f in frames] if frames else None),
        "report": report.model_dump() if report else None,
    }


def save_report(run_dir: Path, data: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                      default=str),
                           encoding="utf-8")
    return report_path


def main() -> None:
    # .env 只放密钥类环境变量；已存在的环境变量优先于 .env 中的值
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    try:
        settings = load_settings(args)
    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    check_env(args, settings)

    run_dir = settings.out_dir / args.video.stem
    frames_dir = run_dir / "frames"

    prompt = settings.prompt
    if settings.prompt_file is not None:
        prompt = settings.prompt_file.read_text(encoding="utf-8").strip()

    # 第一阶段：VBench 初筛（本地模型，不花 API 费用）
    prescreen: PrescreenResult | None = None
    if settings.prescreen.enabled:
        ps = settings.prescreen
        print(f"正在运行 VBench 初筛（{len(ps.dims)} 个维度，首次运行需下载模型权重）...")
        prescreen = run_prescreen(args.video, dims=ps.dims,
                                  thresholds=ps.thresholds or None,
                                  timeout=ps.timeout,
                                  python=ps.vbench_python)
        print_prescreen(prescreen)
        if prescreen.status == "failed" and not ps.force:
            report_path = save_report(run_dir, build_report_data(
                args, settings, prompt, prescreen, verdict="prescreen_failed"))
            print(f"\n初筛不通过（低于阈值: {', '.join(prescreen.failed_dims)}），"
                  "已跳过 Claude 检测（--force 可强制继续）。")
            print(f"报告已保存: {report_path}")
            sys.exit(2)

    # 第二阶段：抽帧
    print(f"\n正在从 {args.video.name} 抽取 {settings.frames} 帧 ...")
    try:
        frames = extract_frames(args.video, frames_dir,
                                n=settings.frames, seed=settings.seed)
    except FrameExtractionError as e:
        print(f"抽帧失败: {e}", file=sys.stderr)
        sys.exit(1)
    for f in frames:
        print(f"  帧 #{f.index}  t={f.timestamp:.2f}s  -> {f.path}")

    # 第三阶段：Claude 检测
    print(f"\n正在调用 Claude API（{settings.model}）检测 ..."
          + ("（已提供生成 prompt）" if prompt else ""))
    try:
        report = run_qc(frames, args.ref, model=settings.model, prompt=prompt)
    except anthropic.AuthenticationError:
        print("Claude API 认证失败：请检查 .env 中的 ANTHROPIC_API_KEY 是否有效",
              file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("无法连接 Claude API：请检查网络（及 ANTHROPIC_BASE_URL 配置）",
              file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"Claude API 调用失败（HTTP {e.status_code}）: {e.message}",
              file=sys.stderr)
        sys.exit(1)

    print_summary(report, frames)

    report_path = save_report(run_dir, build_report_data(
        args, settings, prompt, prescreen, frames=frames, report=report))
    print(f"完整报告已保存: {report_path}")
    print(f"抽帧图片目录: {frames_dir}")


if __name__ == "__main__":
    main()
