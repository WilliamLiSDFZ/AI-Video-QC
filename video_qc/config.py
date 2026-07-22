"""运行配置：YAML 配置文件（可选）+ CLI 覆盖。

优先级：CLI 显式传参 > --config YAML > 内置默认值。
video 与参照图不属于配置，必须由命令行传入。
密钥类（ANTHROPIC_API_KEY 等）不放这里，放 .env / 环境变量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .claude_checker import DEFAULT_MODEL
from .prescreener import ALL_DIMS


class ConfigError(ValueError):
    pass


@dataclass
class PrescreenSettings:
    enabled: bool = True
    force: bool = False
    dims: list[str] = field(default_factory=lambda: list(ALL_DIMS))
    thresholds: dict[str, float] = field(default_factory=dict)  # 覆盖默认阈值
    vbench_python: Path | None = None
    timeout: int = 3600


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    frames: int = 5
    seed: int | None = None
    out_dir: Path = Path("output")
    prompt: str | None = None
    prompt_file: Path | None = None
    prescreen: PrescreenSettings = field(default_factory=PrescreenSettings)
    config_path: Path | None = None  # 使用的配置文件，写入报告溯源


_TOP_KEYS = {"model", "frames", "seed", "out_dir", "prompt", "prompt_file", "prescreen"}
_PRESCREEN_KEYS = {"enabled", "force", "dims", "thresholds", "vbench_python", "timeout"}


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"配置文件不是合法 YAML: {path}\n{e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是键值映射: {path}")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"配置文件包含未知字段: {', '.join(sorted(unknown))}"
                          f"（可用: {', '.join(sorted(_TOP_KEYS))}）")
    ps = data.get("prescreen") or {}
    if not isinstance(ps, dict):
        raise ConfigError("prescreen 必须是键值映射")
    unknown = set(ps) - _PRESCREEN_KEYS
    if unknown:
        raise ConfigError(f"prescreen 包含未知字段: {', '.join(sorted(unknown))}"
                          f"（可用: {', '.join(sorted(_PRESCREEN_KEYS))}）")
    if data.get("prompt") and data.get("prompt_file"):
        raise ConfigError("prompt 与 prompt_file 只能设置其一")
    return data


def _apply_yaml(s: Settings, data: dict) -> None:
    if "model" in data and data["model"]:
        s.model = str(data["model"])
    if "frames" in data and data["frames"] is not None:
        s.frames = int(data["frames"])
    if "seed" in data and data["seed"] is not None:
        s.seed = int(data["seed"])
    if "out_dir" in data and data["out_dir"]:
        s.out_dir = Path(data["out_dir"])
    if data.get("prompt"):
        s.prompt = str(data["prompt"])
    if data.get("prompt_file"):
        s.prompt_file = Path(data["prompt_file"])
    ps = data.get("prescreen") or {}
    p = s.prescreen
    if "enabled" in ps:
        p.enabled = bool(ps["enabled"])
    if "force" in ps:
        p.force = bool(ps["force"])
    if "dims" in ps and ps["dims"]:
        dims = ps["dims"]
        if isinstance(dims, str):
            dims = [d.strip() for d in dims.split(",") if d.strip()]
        p.dims = [str(d) for d in dims]
    if ps.get("thresholds"):
        p.thresholds = {str(k): float(v) for k, v in ps["thresholds"].items()}
    if ps.get("vbench_python"):
        p.vbench_python = Path(ps["vbench_python"])
    if "timeout" in ps and ps["timeout"] is not None:
        p.timeout = int(ps["timeout"])


def load_settings(args) -> Settings:
    """合并 默认值 → YAML → CLI 后返回最终配置；非法配置抛 ConfigError。"""
    s = Settings()
    if args.config is not None:
        _apply_yaml(s, _load_yaml(args.config))
        s.config_path = args.config

    # CLI 显式传参覆盖（argparse 默认值均为 None / False，据此判断是否显式传入）
    if args.model is not None:
        s.model = args.model
    if args.frames is not None:
        s.frames = args.frames
    if args.seed is not None:
        s.seed = args.seed
    if args.out_dir is not None:
        s.out_dir = args.out_dir
    if args.prompt is not None or args.prompt_file is not None:
        s.prompt, s.prompt_file = args.prompt, args.prompt_file
    if args.no_prescreen:
        s.prescreen.enabled = False
    if args.force:
        s.prescreen.force = True
    if args.prescreen_dims is not None:
        s.prescreen.dims = [d.strip() for d in args.prescreen_dims.split(",")
                            if d.strip()]

    # 合并后统一校验
    if s.frames < 1:
        raise ConfigError(f"frames 必须 ≥ 1，当前为 {s.frames}")
    bad_dims = [d for d in s.prescreen.dims if d not in ALL_DIMS]
    if bad_dims:
        raise ConfigError(f"未知的初筛维度: {', '.join(bad_dims)}"
                          f"（可选: {', '.join(ALL_DIMS)}）")
    bad_th = [d for d in s.prescreen.thresholds if d not in ALL_DIMS]
    if bad_th:
        raise ConfigError(f"thresholds 包含未知维度: {', '.join(bad_th)}")
    return s
