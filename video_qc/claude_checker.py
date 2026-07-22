"""调用 Claude API 对抽帧 + 参照图做质量检测，返回结构化报告。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .frame_extractor import Frame

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """你是一名专业的 AI 生成视频质量审核员。
用户会提供：
1. 参照物的实拍照片（真实外观基准）
2. 从一段 AI 生成的视频中抽取的若干帧（按时间顺序编号）

你的任务是逐帧仔细检查以下问题：
- deformation: 主体形状扭曲、比例失调、结构变形
- anatomy: 人物/动物的肢体、手指、面部等解剖结构异常
- physics: 不符合物理常理的现象（悬浮、穿模、光影矛盾、透视错误等）
- reference_mismatch: 视频中的主体与参照物照片不一致（颜色、材质、logo、细节缺失或多余）
- artifact: AI 生成痕迹（纹理糊化、噪点、文字乱码、边缘融化等）
- other: 其他不合常理之处

要求：
- 逐帧给出发现的问题；某帧没有问题就返回空的 issues 列表，不要凭空捏造问题
- 描述问题时指明具体位置和表现，便于人工对照抽帧图片复核
- 所有描述用中文
- overall_score 为 0-10 的整数，只评画面生成质量，10 表示完全逼真无缺陷，0 表示严重失真

若用户提供了该视频的生成 prompt：
- 把 prompt 拆解为一条条可独立核对的具体要求（主体、动作、场景、风格、镜头、文字等），逐条判断实现状态：
  - met: 抽帧中可确认已实现
  - partially_met: 部分实现或实现有偏差
  - not_met: 抽帧中可确认未实现
  - cannot_judge: 仅凭静态抽帧无法判断（如动作过程、时序、镜头运动、音频类要求），必须如实标注，禁止猜测
- 结合 prompt 语境判断缺陷：prompt 明确要求的风格化效果（如卡通、夸张色彩）不算 artifact 缺陷；与 prompt 要求相悖之处要指出
- prompt_adherence_score 为 0-10 的整数，只衡量 prompt 要求的实现程度（cannot_judge 的条目不计入扣分），与画面质量分互相独立
若未提供生成 prompt：prompt_requirements 返回空列表，prompt_adherence 与 prompt_adherence_score 返回 null。"""


class Issue(BaseModel):
    category: Literal["deformation", "anatomy", "physics",
                      "reference_mismatch", "artifact", "other"] = Field(
        description="问题类别")
    severity: Literal["low", "medium", "high"] = Field(description="严重程度")
    description: str = Field(description="问题的具体描述（中文），指明位置和表现")


class FrameFinding(BaseModel):
    frame_index: int = Field(description="对应第几帧（从 1 开始）")
    issues: list[Issue] = Field(description="该帧发现的问题，无问题则为空列表")


class RequirementCheck(BaseModel):
    requirement: str = Field(description="从生成 prompt 中拆解出的一条具体要求（中文）")
    status: Literal["met", "partially_met", "not_met", "cannot_judge"] = Field(
        description="实现状态；静态抽帧无法判断的要求（动作过程、时序、音频等）标 cannot_judge")
    note: str = Field(description="判断依据（中文），指明在哪些帧看到了什么")


class QCReport(BaseModel):
    frame_findings: list[FrameFinding] = Field(description="逐帧检测结果，每帧一项")
    reference_consistency: str = Field(
        description="视频主体与参照物照片一致性的总体评估（中文）")
    overall_assessment: str = Field(description="视频整体生成质量的总体评价（中文）")
    overall_score: int = Field(
        description="0-10 的整数画面质量分，10 为完全逼真无缺陷")
    has_defects: bool = Field(description="是否发现任何缺陷")
    prompt_requirements: list[RequirementCheck] = Field(
        description="生成 prompt 拆解出的要求及逐条实现状态；未提供 prompt 时为空列表")
    prompt_adherence: str | None = Field(
        description="prompt 要求实现情况的总体评估（中文）；未提供 prompt 时为 null")
    prompt_adherence_score: int | None = Field(
        description="0-10 的整数 prompt 实现度分数；未提供 prompt 时为 null")


def _image_block(path: Path) -> dict:
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        raise ValueError(f"不支持的图片格式: {path}（支持 jpg/png/gif/webp）")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def run_qc(frames: list[Frame], refs: list[Path],
           model: str = DEFAULT_MODEL, prompt: str | None = None) -> QCReport:
    content: list[dict] = []
    if prompt:
        content.append({
            "type": "text",
            "text": f"该视频由以下生成 prompt 产生（原文）：\n<generation_prompt>\n{prompt}\n</generation_prompt>",
        })
    for i, ref in enumerate(refs, 1):
        content.append({"type": "text", "text": f"参照物实拍照片 #{i}:"})
        content.append(_image_block(ref))
    for frame in frames:
        content.append({
            "type": "text",
            "text": f"视频抽帧 #{frame.index}（时间点 {frame.timestamp:.1f} 秒）:",
        })
        content.append(_image_block(frame.path))
    instruction = ("请对照参照物照片，逐帧检测以上视频抽帧是否存在变形或不符合常理之处，"
                   "输出结构化报告。")
    if prompt:
        instruction += ("同时把生成 prompt 拆解为逐条要求，判断每条的实现状态，"
                        "并结合 prompt 语境判断哪些表现属于缺陷。")
    content.append({"type": "text", "text": instruction})

    client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL 等
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_format=QCReport,
    )
    report = response.parsed_output
    if report is None:
        raise RuntimeError(
            f"模型未返回合法的结构化报告（stop_reason={response.stop_reason}）")
    return report
