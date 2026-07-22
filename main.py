#!/usr/bin/env python3
"""AI 生成视频质量检测 baseline。

给定一段 AI 生成的视频和参照物实拍照片，随机抽取若干帧，
调用 Claude API 检测变形、不合常理及与参照物不一致之处。

用法:
    python main.py --video demo.mp4 --ref ref.jpg [ref2.jpg ...] [--frames 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from video_qc.claude_checker import DEFAULT_MODEL, QCReport, run_qc
from video_qc.frame_extractor import Frame, FrameExtractionError, extract_frames

SEVERITY_MARK = {"low": "·", "medium": "⚠", "high": "✖"}
STATUS_MARK = {"met": "✓", "partially_met": "◐", "not_met": "✗", "cannot_judge": "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 生成视频质量检测 baseline")
    parser.add_argument("--video", required=True, type=Path, help="AI 生成的视频文件")
    parser.add_argument("--ref", required=True, type=Path, nargs="+",
                        help="参照物实拍照片（可多张）")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None,
                              help="生成该视频所用的原始 prompt 文本")
    prompt_group.add_argument("--prompt-file", type=Path, default=None,
                              help="从文件读取原始 prompt（utf-8）")
    parser.add_argument("--frames", type=int, default=5, help="抽帧数量（默认 5）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，用于复现抽帧位置")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude 模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--out-dir", type=Path, default=Path("output"),
                        help="输出目录（默认 output/）")
    return parser.parse_args()


def check_env(args: argparse.Namespace) -> None:
    errors = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        errors.append("未找到 ffmpeg/ffprobe，请先安装: brew install ffmpeg")
    if not args.video.is_file():
        errors.append(f"视频文件不存在: {args.video}")
    for ref in args.ref:
        if not ref.is_file():
            errors.append(f"参照图片不存在: {ref}")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        errors.append(f"prompt 文件不存在: {args.prompt_file}")
    has_cred = (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())
    if not has_cred:
        errors.append("未找到 Claude API 凭证，请设置 ANTHROPIC_API_KEY "
                      "（或 ANTHROPIC_AUTH_TOKEN / `ant auth login`）")
    if errors:
        for e in errors:
            print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


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


def main() -> None:
    args = parse_args()
    check_env(args)

    run_dir = args.out_dir / args.video.stem
    frames_dir = run_dir / "frames"

    print(f"正在从 {args.video.name} 抽取 {args.frames} 帧 ...")
    try:
        frames = extract_frames(args.video, frames_dir,
                                n=args.frames, seed=args.seed)
    except FrameExtractionError as e:
        print(f"抽帧失败: {e}", file=sys.stderr)
        sys.exit(1)
    for f in frames:
        print(f"  帧 #{f.index}  t={f.timestamp:.2f}s  -> {f.path}")

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    print(f"\n正在调用 Claude API（{args.model}）检测 ..."
          + ("（已提供生成 prompt）" if prompt else ""))
    report = run_qc(frames, args.ref, model=args.model, prompt=prompt)

    print_summary(report, frames)

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps({
        "video": str(args.video),
        "references": [str(r) for r in args.ref],
        "generation_prompt": prompt,
        "model": args.model,
        "seed": args.seed,
        "frames": [{"index": f.index, "timestamp": f.timestamp,
                    "path": str(f.path)} for f in frames],
        "report": report.model_dump(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整报告已保存: {report_path}")
    print(f"抽帧图片目录: {frames_dir}")


if __name__ == "__main__":
    main()
