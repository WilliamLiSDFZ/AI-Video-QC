"""用 ffmpeg 从视频中分层随机抽帧。"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FrameExtractionError(RuntimeError):
    pass


@dataclass
class Frame:
    index: int  # 从 1 开始
    timestamp: float  # 秒
    path: Path


def get_duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FrameExtractionError(f"ffprobe 读取时长失败: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise FrameExtractionError(f"无法解析视频时长: {result.stdout!r}")


def extract_frames(video: Path, out_dir: Path, n: int = 5,
                   seed: int | None = None) -> list[Frame]:
    """把视频时长均分为 n 段，每段内随机取一个时间点抽一帧。

    分层随机保证随机性的同时覆盖全片，避免纯随机时多帧挤在同一片段。
    """
    duration = get_duration(video)
    # 首尾各留余量，避开片头黑帧和末尾 seek 不到帧的问题
    margin = min(0.2, duration / 10)
    usable = duration - 2 * margin
    if usable <= 0 or duration / n < 0.05:
        raise FrameExtractionError(f"视频过短 ({duration:.2f}s)，无法抽取 {n} 帧")

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = usable / n
    frames: list[Frame] = []
    for i in range(n):
        t = margin + seg * i + rng.uniform(0, seg)
        path = out_dir / f"frame_{i + 1:02d}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "2", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not path.exists():
            raise FrameExtractionError(
                f"抽取第 {i + 1} 帧 (t={t:.2f}s) 失败: {result.stderr.strip()}")
        frames.append(Frame(index=i + 1, timestamp=round(t, 3), path=path))
    return frames
