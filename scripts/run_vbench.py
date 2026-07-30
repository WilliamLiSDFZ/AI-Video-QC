#!/usr/bin/env python
"""在独立的 VBench 环境中运行，由主管线通过 subprocess 调用。

本脚本不依赖主项目代码，只依赖 VBench 环境（torch + vbench）。
输出标准化 JSON 到 --out 文件：
    {"device": "...", "scores": {维度: 分数}, "errors": {维度: 错误信息}}
退出码：只要有任一维度产出分数即为 0；全部失败为 1。
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="待评测的视频文件")
    parser.add_argument("--dims", required=True, help="逗号分隔的 VBench 维度名")
    parser.add_argument("--out", required=True, help="结果 JSON 输出路径")
    args = parser.parse_args()

    import torch

    # PyTorch 2.6 起 torch.load 默认 weights_only=True，而 VBench 的 AMT
    # （motion_smoothness）等旧权重里含 typing.OrderedDict 这类非白名单全局对象，
    # 加载会直接 UnpicklingError。这些权重由 VBench 自己从官方地址下载到
    # ~/.cache/vbench，来源可信，故在这个专跑 VBench 的子进程里把默认值改回去。
    # 用 setdefault：显式传了 weights_only 的调用方仍按其意愿执行。
    _torch_load = torch.load

    def _torch_load_compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load(*a, **kw)

    torch.load = _torch_load_compat

    import vbench as vbench_pkg
    from vbench import VBench

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_info = Path(vbench_pkg.__file__).resolve().parent / "VBench_full_info.json"

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        vb = VBench(device, str(full_info), tmp)
        for dim in dims:
            # 每个维度单独评测、单独容错：个别维度在 CPU 上不兼容或权重
            # 下载失败时，不影响其余维度出分
            try:
                vb.evaluate(
                    videos_path=args.video,
                    name=f"qc_{dim}",
                    dimension_list=[dim],
                    mode="custom_input",
                )
                result_file = Path(tmp) / f"qc_{dim}_eval_results.json"
                data = json.loads(result_file.read_text(encoding="utf-8"))
                scores[dim] = float(data[dim][0])
            except Exception as e:  # noqa: BLE001
                errors[dim] = f"{type(e).__name__}: {e}"

    Path(args.out).write_text(
        json.dumps({"device": str(device), "scores": scores, "errors": errors},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sys.exit(0 if scores else 1)


if __name__ == "__main__":
    main()
