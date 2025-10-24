from __future__ import annotations
import io
import json
import os
import uuid
import datetime
from typing import List, Optional, Union, Literal

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from PIL import Image, ImageOps

from simple_rag import mrag_run


# =========================
# Pydantic v2 数据模型（与前端 HazardData 对齐）
# =========================

class KBCitation(BaseModel):
    doc_id: str
    title: Optional[str] = None
    clause: Optional[str] = None
    url: Optional[str] = None


class EvidenceRegionGlobal(BaseModel):
    # 用 Literal 取代 v1 的 const
    type: Literal["global"] = "global"


class EvidenceRegionBBox(BaseModel):
    type: Literal["bbox_xywh_norm"] = "bbox_xywh_norm"
    bbox: List[float]  # [x, y, w, h], 归一化 0-1


EvidenceRegion = Union[EvidenceRegionGlobal, EvidenceRegionBBox]


class HazardItem(BaseModel):
    id: str
    category: str
    label: str
    evidence_region: EvidenceRegion
    evidence_quality: Optional[str] = "unknown"
    kb_citations: Optional[List[KBCitation]] = []
    severity: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SceneGuess(BaseModel):
    scene: str
    rationale: Optional[str] = None


class HazardData(BaseModel):
    task_id: str
    image_path: Optional[str] = None
    scene_guess: Optional[SceneGuess] = None
    auto_queries: Optional[List[str]] = []
    overall_risk_score: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    hazards: List[HazardItem]

    @field_validator("hazards")
    @classmethod
    def hazards_is_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("hazards must be a list")
        return v


# =========================
# FastAPI 初始化 & CORS
# =========================

app = FastAPI(title="Construction Hazard Analyzer API", version="0.2.0")

# 生产环境请收紧 allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态上传目录（供前端访问）
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 将 /uploads 挂载为静态路径，前端可用 <img src="/uploads/xxx.png">
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# =========================
# 工具函数
# =========================

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def save_image_and_fix_orientation(file_bytes: bytes, original_name: str) -> str:
    """
    读取图片 -> 纠正 EXIF 方向 -> 保存为 PNG 以 UUID 命名。
    返回前端可直接访问的相对路径（/uploads/xxx.png）。
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)

        if img.width < 32 or img.height < 32:
            raise ValueError("Image is too small.")

        out_name = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        img.save(out_path, format="PNG")

        # 返回可被静态服务访问的路径
        return f"D:/LLM/project/ustarconstruct/uploads/{out_name}"

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )


def try_parse_json_or_jsonl_firstline(bytes_data: bytes) -> dict:
    """
    优先整体 JSON，若失败则解析 JSONL 的第一行。
    """
    text = bytes_data.decode("utf-8", errors="ignore").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty JSON/JSONL content."
        )

    # 尝试整体 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试第一行 JSONL
    first_line = text.splitlines()[0]
    try:
        return json.loads(first_line)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse JSON/JSONL: {str(e)}"
        )


# =========================
# 伪分析逻辑（可替换为真实 VLM 推理）
# =========================

def analyze_image(saved_path_for_frontend: str) -> HazardData:
    """
    返回一个演示的 HazardData。
    你可以替换为真实的模型推理，并组装为此结构。
    """
    task_id = uuid.uuid4().hex
    demo = HazardData(
        task_id=task_id,
        image_path=saved_path_for_frontend,
        scene_guess=SceneGuess(
            scene="吊装指挥",
            rationale="图中可见起重设备与悬挂重物，地面作业人员疑似进入回转半径"
        ),
        auto_queries=[
            "吊装作业 警戒线 缺失 规范 条款",
            "起重指挥 信号 不明确 标准",
            "回转半径 人员入内 风险 防控",
            "吊索具 报废标准 检查要点",
            "立体交叉作业 隔离 措施"
        ],
        overall_risk_score=0.62,
        hazards=[
            HazardItem(
                id="hazard_001",
                category="吊装作业",
                label="吊装作业区未设警戒线，存在人员误入回转半径风险",
                evidence_region=EvidenceRegionGlobal(),
                evidence_quality="clear",
                kb_citations=[
                    KBCitation(
                        doc_id="JGJ 59-2011",
                        title="建筑施工安全检查标准",
                        clause="起重吊装作业区应设置警戒范围并设明显标识",
                        url="https://example.org/specs/JGJ59"
                    )
                ],
                severity="中等",
                confidence=0.78
            ),
            HazardItem(
                id="hazard_002",
                category="PPE",
                label="个别人员疑似未正确佩戴安全帽",
                evidence_region=EvidenceRegionBBox(bbox=[0.32, 0.18, 0.12, 0.20]),
                evidence_quality="uncertain",
                kb_citations=[],
                severity="较低",
                confidence=0.55
            ),
        ]
    )
    return demo


# =========================
# 路由：文件上传与分析
# =========================

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    处理前端 <input type=file> 上传：
    - image/* ：保存图片并分析，返回 HazardData
    - application/json / .jsonl ：解析并按 HazardData 校验后返回
    统一错误返回 {"detail": "..."} 以适配前端捕获。
    """
    try:
        content_type = (file.content_type or "").lower()
        filename = (file.filename or "").lower()
        data = await file.read()

        # ---- 图片 ----
        if content_type.startswith("image/") or any(
            filename.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        ):
            saved_visual_path = save_image_and_fix_orientation(data, file.filename)
            print(saved_visual_path)
            result = mrag_run(saved_visual_path)
            return JSONResponse(status_code=200, content=result)

        # ---- JSON / JSONL ----
        if (
            content_type in ("application/json", "application/x-ndjson")
            or filename.endswith(".json")
            or filename.endswith(".jsonl")
        ):
            raw_obj = try_parse_json_or_jsonl_firstline(data)
            try:
                hd = HazardData.model_validate(raw_obj)  # v2 写法
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON structure for HazardData: {str(e)}"
                )
            return JSONResponse(status_code=200, content=hd.model_dump())

        # ---- 其他类型 ----
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload an image (PNG/JPG) or a data file (JSON/JSONL).",
        )

    except HTTPException as he:
        # 显式业务错误
        raise he
    except Exception as e:
        # 未知错误
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# =========================
# 健康检查
# =========================

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": _now_iso()}
