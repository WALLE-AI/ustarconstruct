"""
Safety Compliance RAG Pipeline (Image-only, V2)
================================================

Upgraded end‑to‑end pipeline per user spec:
1) 输入: 图像 + 隐患类别配置 `hazard_cfg`
2) 使用 VLM 进行 场景判断 + 隐患识别（可视化证据 bbox/quality/confidence）
3) 基于【场景 + 每条隐患的描述/标签】自动构造检索 Query → 规范库检索 (`kb.search`)
4) 汇总条款片段（去重/排序/裁剪）
5) 交由 VLM 执行 **单次整体合成**： (系统契约 + 场景猜测 + 自动检索片段 + 原图) → **最终 JSON**
6) 如 VLM 不可用/失败，降级为程序化组装

Contract:
- Input: image path or PIL.Image (RGB) + HazardConfig
- Output: JSON object strictly following SCHEMA_TEXT；末尾追加 ≤120 字中文小结
- No reasoning chain in the output; only structured fields + short Chinese summary.

KB Contract (duck-typed):
    search(query: str, top_k: int = 5) -> {"results": [{
        "doc_id": str, "title": str, "clause_id": str,
        "clause_text": str, "relevance": float
    }]}

VLM Contract (duck-typed):
    infer_scene(image) -> {"scene": str, "rationale": str}
    detect_hazards(image, hazard_cfg) -> List[{
        "category": str, "label": str, "bbox_xywh_norm": [x,y,w,h],
        "evidence_quality": "clear|blurry|occluded|low_res", "confidence": float
    }]
    render_final(image, scene_guess, auto_queries, kb_snippets, schema_contract) -> Dict

Notes:
- 提供 DummyKB / DummyVLM 以便在无依赖环境可直跑
- 提供最小的 HazardConfig 数据结构
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union
import base64, mimetypes
from io import BytesIO
from typing import Union
from PIL import Image

from backend.agent.kb_search import KBSimpleSearch
from dotenv import load_dotenv
load_dotenv()

try:
    from PIL import Image
except Exception:  # pillow is optional for headless usage
    Image = None  # type: ignore

# -----------------------------
# Types & Data Structures
# -----------------------------
BBox = Tuple[float, float, float, float]  # (x, y, w, h) normalized

@dataclass
class Citation:
    doc_id: str
    clause_id: str
    clause_brief: str
    match_type: str  # "direct" | "analogous" | "NO_KB_MATCH"

@dataclass
class HazardProposal:
    category: str
    label: str
    bbox_xywh_norm: List[float]
    evidence_quality: str  # "clear" | "blurry" | "occluded" | "low_res"
    confidence: float

@dataclass
class HazardCfgItem:
    key: str  # 如 "临边防护"
    category: str  # 如 "临边"
    # 用于构造检索 Query 的关键词模板（object × behavior × consequence）
    # 例如: ["{scene} 防护栏 缺失 坠落 规范 条款", "临边 防护栏 缺失 坠落"]
    query_templates: List[str] = field(default_factory=list)
    # 别名/同义词（可用于粗匹配 label）
    synonyms: List[str] = field(default_factory=list)

@dataclass
class HazardConfig:
    items: List[HazardCfgItem] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HazardConfig":
        items = []
        for obj in d.get("items", []):
            items.append(HazardCfgItem(
                key=obj.get("key", ""),
                category=obj.get("category", "综合"),
                query_templates=list(obj.get("query_templates", [])),
                synonyms=list(obj.get("synonyms", [])),
            ))
        return HazardConfig(items=items)

    def match_templates(self, label: str, scene_tail: str) -> List[str]:
        """根据 label 模糊匹配到配置项，返回合成后的查询模板列表。"""
        ls = label.lower()
        out: List[str] = []
        for it in self.items:
            if it.key and it.key.lower() in ls:
                out.extend(_materialize_templates(it.query_templates, scene_tail))
            else:
                for syn in it.synonyms:
                    if syn.lower() in ls:
                        out.extend(_materialize_templates(it.query_templates, scene_tail))
                        break
        return out

# -----------------------------
# KB Client (duck-typed) & DummyKB
# -----------------------------
class KBClient:
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        search = KBSimpleSearch()
        return search.kb_search(query=query,top_k=top_k)

class DummyKB(KBClient):
    _DATA = [
        {
            "doc_id": "GB-50870-2013",
            "title": "建筑施工安全检查标准",
            "clause_id": "6.2.1",
            "clause_text": "临边作业应设置两道防护栏杆及踢脚板，缺失属重大隐患。",
            "relevance": 0.92,
        },
        {
            "doc_id": "JGJ59-2011",
            "title": "建筑施工扣件式钢管脚手架安全技术规范",
            "clause_id": "4.1.2",
            "clause_text": "脚手架作业人员应佩戴安全帽、安全带；缺失属于较大隐患。",
            "relevance": 0.88,
        },
        {
            "doc_id": "GB-50140-2005",
            "title": "建筑灭火器配置设计规范",
            "clause_id": "5.1.3",
            "clause_text": "疏散通道应保持畅通，不得堆放物料阻塞。",
            "relevance": 0.86,
        },
        {
            "doc_id": "GB-16895.1-2002",
            "title": "低压配电装置和系统",
            "clause_id": "7.3",
            "clause_text": "可触及带电部分需防护隔离与标识。",
            "relevance": 0.84,
        },
    ]

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = []
        for r in self._DATA:
            score = r["relevance"]
            overlap = sum(1 for w in ["临边","防护栏","坠落","疏散","通道","带电","安全帽"] if w in query)
            score = min(0.99, score + 0.01 * overlap)
            hits.append({**r, "relevance": score})
        hits.sort(key=lambda x: x["relevance"], reverse=True)
        return {"results": hits[:top_k]}
    

def _image_to_data_url(image: Union[str, Image.Image]) -> str:
    """
    将本地路径或 PIL.Image 转为 data:image/...;base64,xxx
    便于以 'input_image' 传入 Responses API
    """
    if isinstance(image, str):
        mime, _ = mimetypes.guess_type(image)
        mime = mime or "image/jpeg"
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    elif isinstance(image, Image.Image):
        mime = "image/png"
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    else:
        raise ValueError("image must be a filepath or PIL.Image")

# -----------------------------
# VLM Hook Contracts & Dummies
# -----------------------------
class VLMClient:
    def infer_scene(self, image: Union[str, Image.Image]) -> Dict[str, str]:
        raise NotImplementedError

    def detect_hazards(self, image: Union[str, Image.Image], hazard_cfg: HazardConfig) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def render_final(self,
                     image: Union[str, Image.Image],
                     scene_guess: Dict[str, str],
                     auto_queries: List[str],
                     kb_snippets: List[Dict[str, Any]],
                     schema_contract: str) -> Dict[str, Any]:
        raise NotImplementedError
    
def generate_queries(self, image: Union[str, Image.Image], 
                     scene_guess: Dict[str,str],
                     proposals: List[Dict[str, Any]], 
                     hazard_cfg: HazardConfig,
                     max_queries: int = 5) -> List[str]: 
    raise NotImplementedError
    
import httpx
import openai

class LLMApi():
    def __init__(self) -> None:
        self.des = "llm api service"
    def __str__(self) -> str:
        return self.des
    
    def init_client_config(self,llm_type):
        if llm_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key  = os.environ.get("OPENROUTER_API_KEY")
        elif llm_type =='siliconflow':
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        elif llm_type=="openai":
            base_url = "https://api.openai.com/v1"
            api_key  = os.environ.get("OPENAI_API_KEY")
        elif llm_type == "starvlm":
            base_url = os.getenv("VLM_SERVE_HOST")
            api_key  = "empty"
        elif llm_type == "local":
            base_url = os.getenv("LOCAL_BASE_URL")
            api_key  = "empty"
            
        return base_url,api_key     
    @classmethod
    def llm_client(cls,llm_type):
        base_url,api_key = cls().init_client_config(llm_type)
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client
        
client = LLMApi.llm_client(llm_type="local")

class OpenAIVLM(VLMClient):
    """
    用 OpenAI Responses API 实现:
      - infer_scene: 场景 + 简短理由
      - detect_hazards: 按你的 hazard_cfg 进行识别，输出结构化 proposals
      - render_final: 一次生成最终 JSON（遵循 schema_contract）
    """

    model = "Starvlm-8b-instruct"  # 可按需替换为企业侧可用的多模态模型

    # ----------- Prompt 片段（中文，高质量可直接用）-----------
    SYS_SCENE = (
        "你是安全合规审查专家。仅根据图片判断最匹配场景，"
        "从集合中严格选一：施工现场/临边、电气、起重、消防、PPE、交通、物料堆放、脚手架、基坑、"
        "高处作业、焊接/热作业、密闭空间、吊装指挥、临时用电、疏散/通道、桥梁/钢结构/高强螺栓等等"
        "请基于这张图片判断最贴切的【场景/工况】路径，输出JSON：{\"scene\":<集合中选一>,\"rationale\":<<=40字 并给出≤40字中文理由（只保留关键信息，不要推理链）。>}; 只输出JSON。"
    )

    # USR_SCENE = (
    #     "请基于这张图片判断最贴切的【场景/工况】路径，如：施工现场/临边、施工现场/电气/临时用电、桥梁/钢结构/高强螺栓等；"
    #     "并给出≤40字中文理由（只保留关键信息，不要推理链）。"
    # )

    SYS_DET = (
        "你是一名图像安全/质量隐患检测员。仅凭单张图片，识别**可见**隐患并给出证据框。\n"
        "如果置信不足或画质差，可将 evidence_region.type 设为 \"global\"；"
        "否则输出归一化 bbox（x,y,w,h ∈[0,1]）。\n"
        "标签优先从给定的 labels 里选，若无精确匹配，可用其同义词（synonyms）或等价表达。\n"
        "严格输出 JSON 数组，每个元素形如：{\n"
        "  \"category\": \"<来自类别名称>\",\n"
        "  \"label\": \"<来自 labels 的最贴切项或其等价表达>\",\n"
        "  \"bbox_xywh_norm\": [x,y,w,h],   # 若不可框定则用 [0,0,0,0] 并在下方指明 global\n"
        "  \"evidence_quality\": \"clear|blurry|occluded|low_res\",\n"
        "  \"confidence\": <0-1 浮点>\n"
        "}\n"
        "只输出 JSON，不要解释。"
    )

    DET_INST_TPL = (
        "场景提示：{scene_path}\n"
        "请仅在【下列候选类别】内进行识别（可 0~N 条）：\n"
        "{cat_list}\n"
        "说明：每个类别含 `labels`（优先使用）与 `synonyms`（可模糊匹配）；"
        "若无法确认，请给 global 证据、降低 confidence。"
    )

    SYS_FINAL = (
        "你是一名安全合规审查专家。请基于：图片 + 场景猜测 + 自动检索得到的规范条款片段 + 结构化契约，"
        "一次性输出**最终合规 JSON**并满足契约字段，禁止输出推理链或多余文本。"
    )
    
    SYS_QGEN = (
        "你是一名面向工程规范检索的 Query 生成工程师。"
        "目标：基于【场景 + 隐患候选】生成 3–5 条中文检索词。"
        "每条应包含：<对象/构件> + <违规/状态> + <可能后果>。"
        "要求：① 去重不近义；② 贴近条款用语（临边/栏杆/踢脚板/漏保/接地…）；③ 严格 JSON 输出 {\"queries\":[...]}，禁止解释。"
    )
    QGEN_INST_TPL = (
        "【场景】{scene_path}"
        "【隐患候选】{hazards}"
        "【类别参考（仅供理解，勿逐字照抄）】{cat_refs}"
        "请综合以上信息输出 3–5 条高质量 Query。"
        )

    FINAL_INST_TPL = (
        "【场景猜测】{scene_guess}\n"
        "【自动检索 Query】{queries}\n"
        "【规范条款片段（去重排序后取前若干）】\n"
        "{kb_snippets}\n"
        "【输出契约（务必严格遵守字段/枚举/bbox范围）】\n"
        "{schema_text}\n"
        "注意：\n"
        "1) hazards.*.kb_citations 至少给 1 条来自上方片段；\n"
        "2) 末尾追加≤120字中文小结（允许在 JSON 之后紧跟一行小结文本，或放入单独字段 summary）。"
    )
    def generate_queries(self, image, scene_guess, proposals, hazard_cfg, max_queries: int = 5) -> List[str]:
        img_data = _image_to_data_url(image)
        # 组装输入文本（控长）
        hz_lines = []
        for pr in proposals[:6]:
            hz_lines.append(f"- {pr.get('category','综合')} | {pr.get('label','疑似隐患')} | conf={float(pr.get('confidence',0)):.2f}")
        hazards_text = ".join(hz_lines)" or "(无候选)"
        cat_lines = []
        # for it in (hazard_cfg.items or [])[:20]:
        #     cat_lines.append(f"- {it.key}: labels={','.join(it.labels[:8])}; syn={','.join(it.synonyms[:8])}")
        # cat_refs = "".join(cat_lines) or "(无)"
        for key,it in hazard_cfg["categories"].items():
            cat_lines.append(f"- 类别: {key}（kind={it["kind"]}）\n  labels: {', '.join(it["labels"][:12])}\n  synonyms: {', '.join(it["synonyms"][:12])}")
        cat_block = "\n".join(cat_lines[:24])  # 避免超长
        inst = self.QGEN_INST_TPL.format(
            scene_path=scene_guess.get("scene","施工现场/综合"),
            hazards=hazards_text,
            cat_refs=cat_block,
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":[{"type":"text","text": self.SYS_QGEN}]},
                {"role":"user","content":[
                    {"type":"text","text": inst},
                    {"type":"image_url","image_url":{"url": img_data}},
                ]}
            ]
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            qs_raw = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
            return qs_raw
        except Exception as e:
            print("e:",e)
    
    

    def infer_scene(self, image):
        img_data = _image_to_data_url(image)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": [{"type": "text", "text": self.SYS_SCENE}]
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": "给出json"},
                    {"type": "image_url", "image_url": {"url": img_data}}
                ]
            }]
        )
        txt = resp.choices[0].message.content # SDK 会拼接文本输出
        try:
            data = json.loads(txt)
            # 兜底修正
            scene = data.get("scene") or "施工现场/综合"
            rationale = (data.get("rationale") or "图像信息有限")[:40]
            return {"scene": scene, "rationale": rationale}
        except Exception:
            return {"scene": "施工现场/综合", "rationale": "图像信息有限"}

    def detect_hazards(self, image, hazard_cfg: HazardConfig):
        """
        将 YAML categories 编成“可选类别清单”，作为模型识别限定。
        """
        img_data = _image_to_data_url(image)

        # 把 categories 填入提示（只含名称、labels、synonyms，避免过长）
        lines = []
        for key,it in hazard_cfg["categories"].items():
            lines.append(f"- 类别: {key}（kind={it["kind"]}）\n  labels: {', '.join(it["labels"][:12])}\n  synonyms: {', '.join(it["synonyms"][:12])}")
        cat_block = "\n".join(lines[:24])  # 避免超长

        inst = self.DET_INST_TPL.format(scene_path="(未知/待判定)", cat_list=cat_block)

        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": [{"type": "text", "text": self.SYS_DET}]
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": inst},
                    {"type": "image_url", "image_url": {"url": img_data}}
                ]
            }]
        )
        txt = resp.choices[0].message.content # SDK 会拼接文本输出
        try:
            data = json.loads(txt)
            # 支持两种返回：直接数组 或 {"items":[...]}
            items = data if isinstance(data, list) else data.get("items", [])
        except Exception:
            items = []

        # 轻度清洗/裁剪
        cleaned = []
        for it in items[:12]:
            bb = [float(x) for x in (it.get("bbox_xywh_norm") or [0,0,0,0])][:4]
            bb = [max(0.0, min(1.0, v)) for v in (bb+[0,0,0,0])[:4]]
            cleaned.append({
                "category": it.get("category") or "综合",
                "label": it.get("label") or "疑似隐患",
                "bbox_xywh_norm": bb,
                "evidence_quality": it.get("evidence_quality") or "low_res",
                "confidence": float(it.get("confidence") or 0.3),
            })
        return cleaned

    def render_final(self, image, scene_guess, auto_queries, kb_snippets, schema_contract):
        """
        一次成型：把契约 + 场景猜测 + 查询 + 片段 + 图像 一并送入。
        """
        img_data = _image_to_data_url(image)

        kb_lines = []
        for s in kb_snippets[:12]:
            kb_lines.append(f"- [{s.get('doc_id')}] {s.get('clause_id')} | {s.get('clause_text')}")
        kb_text = "\n".join(kb_lines) or "(无检索结果)"

        inst = self.FINAL_INST_TPL.format(
            scene_guess=json.dumps(scene_guess, ensure_ascii=False),
            queries=json.dumps(auto_queries, ensure_ascii=False),
            kb_snippets=kb_text,
            schema_text=schema_contract
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": [{"type": "text", "text": self.SYS_FINAL}]
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": inst},
                    {"type": "image_url", "image_url": {"url": img_data}}
                ]
            }]
        )
        txt = resp.choices[0].message.content # SDK 会拼接文本输出
        try:
            out = json.loads(txt)
        except Exception:
            # 兜底降级，避免调用方崩溃
            out = {
                "task_id": str(uuid.uuid4()),
                "scene_guess": scene_guess,
                "auto_queries": auto_queries,
                "overall_risk_score": 0,
                "hazards": [],
                "missing_info": ["未能按契约解析模型输出，请回退程序化路径或重试"],
                "version": "img-only-v1.2",
            }
        return out

# -----------------------------
# Utils
# -----------------------------

def _open_image(image: Union[str, Image.Image]) -> Optional[Image.Image]:
    if Image is None:
        return None
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    return None

_DEF_GENERIC_QUERIES = [
    "临边 防护栏 缺失 坠落 规范 条款",
    "PPE 安全帽 未佩戴 头部伤害 规范 条款",
    "消防 疏散通道 堵塞 逃生受阻 规范 条款",
]

def _materialize_templates(templates: List[str], scene_tail: str) -> List[str]:
    out = []
    for t in templates:
        out.append(t.replace("{scene}", scene_tail))
    return out

# -----------------------------
# Retrieval helpers
# -----------------------------

def format_kb_context(all_results: Dict[str, List[Dict[str, Any]]], max_items: int = 12) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for rs in all_results.values():
        pool.append(rs)
    uniq = []
    for r in pool:
        docs,distances = r['documents'],r["distances"]
        for doc,dis in zip(docs[0],distances[0]):
            data = json.loads(doc)
            uniq.append({
                "doc_id": data['file_name'],
                "clause_id": data['sub_title'],
                "title": data['title'],
                "clause_text": data['sub_text'],
                "relevance": dis
        })
        if len(uniq) >= max_items:
            break
    return uniq

# -----------------------------
# Risk scoring (used in fallback path)
# -----------------------------

def _estimate_severity_and_factors(label: str, base_conf: float):
    label_s = label.lower()
    if any(k in label_s for k in ["临边", "坠落", "带电", "裸露电"]):
        severity = "A"; factors = (4, 4, 5)
    elif any(k in label_s for k in ["起重", "超载", "脚手架", "关键构件缺失"]):
        severity = "B"; factors = (3, 4, 4)
    elif any(k in label_s for k in ["疏散", "通道", "标识缺失", "堵塞"]):
        severity = "C"; factors = (3, 3, 3)
    else:
        severity = "D"; factors = (2, 2, 2)
    like = max(1, min(5, int(round(factors[0] * (0.6 + 0.8 * base_conf)))))
    return severity, (like, factors[1], factors[2])

def _score_risk(like: int, expo: int, consq: int) -> int:
    return int(round(min(100, like * expo * consq * 4)))

# -----------------------------
# Schema text for prompt injection
# -----------------------------
SCHEMA_TEXT = r"""
仅输出下述 JSON Schema 对象：
{
  "task_id": "<可选：UUID>",
  "scene_guess": {"scene": "<字符串>", "rationale": "<≤40字>"},
  "auto_queries": ["<q1>", "<q2>", "<q3>", "<q4>", "<q5>"],
  "overall_risk_score": 0,
  "hazards": [
    {
      "id": "<string>",
      "category": "<隐患大类>",
      "label": "<具体问题>",
      "evidence_region": {"type": "bbox", "bbox_xywh_norm": [0,0,0,0]} | {"type":"global"},
      "evidence_quality": "<clear|blurry|occluded|low_res>",
      "kb_citations": [{"doc_id":"<标准ID>/文件名称","clause_id":"<条款号>","clause_brief":"<20-40字>","match_type":"<direct|analogous|NO_KB_MATCH>"}],
      "severity": "<A|B|C|D>",
      "risk_score": 0,
      "confidence": 0.0,
      "rationale_brief": "<≤60字>",
      "remediation": ["<立即措施>", "<系统性改进>"],
      "status": "<CONFIRMED|NEEDS_REVIEW>"
    }
  ],
  "summary":"整体总结一下内容，字数≤120字中文小结"
  "missing_info": ["<需要补拍/量化数据>"]，
  "version":"img-only-v1.0"
}
注意：所有 bbox 分量∈[0,1]；每个隐患至少1条来自 kb_snippets 的引用；不要输出推理链。
"""

# -----------------------------
# Orchestrator
# -----------------------------

def analyze_image(
    image: Union[str, Image.Image],
    kb: KBClient,
    hazard_cfg: Optional[HazardConfig] = None,
    task_id: Optional[str] = None,
    vlm: Optional[VLMClient] = None,
) -> Dict[str, Any]:
    """Main entry: 图像 → (VLM 场景+隐患) → RAG → VLM 单次合成（或程序化回退）。"""
    _ = _open_image(image)  # ensure loadable if Pillow present
    vlm = vlm or OpenAIVLM()
    hazard_cfg = hazard_cfg 

    # 1) VLM 场景判断
    scene_guess = {"scene": "施工现场/综合", "rationale": "画面为施工相关"}
    try:
        print("infer scene start")
        s = vlm.infer_scene(image) or {}
        if s.get("scene"):
            scene_guess["scene"] = s["scene"]
        if s.get("rationale"):
            scene_guess["rationale"] = s["rationale"][:40]
    except Exception:
        pass
    scene_tail = scene_guess["scene"].split("/")[-1]

    # 2) VLM 隐患识别（结构化提案）
    proposals_raw: List[Dict[str, Any]] = []
    try:
        print("detect_hazards start")
        proposals_raw = list(vlm.detect_hazards(image, hazard_cfg) or [])
    except Exception as e:
        print("detect hazards:",e)
        proposals_raw = []

    proposals: List[HazardProposal] = []
    for p in proposals_raw:
        bb = p.get("bbox_xywh_norm") or [0,0,0,0]
        bb = [float(max(0.0, min(1.0, float(v)))) for v in (bb + [0,0,0,0])[:4]]
        proposals.append(HazardProposal(
            category=p.get("category", "综合"),
            label=p.get("label", "疑似隐患"),
            bbox_xywh_norm=bb,
            evidence_quality=p.get("evidence_quality", "low_res"),
            confidence=float(p.get("confidence", 0.3)),
        ))

    # 4) 基于【VLM 生成】检索 Query（首选），失败则回退到模板
    auto_queries: List[str] = []
    try:
        print("generate_queries start")
        vlm_qs = vlm.generate_queries(
            image=image,
            scene_guess=scene_guess,
            proposals=[{
                "category": pr.category,
                "label": pr.label,
                "bbox_xywh_norm": pr.bbox_xywh_norm,
                "evidence_quality": pr.evidence_quality,
                "confidence": pr.confidence,
            } for pr in proposals],
            hazard_cfg=hazard_cfg,
            max_queries=5,
        )
        auto_queries = list(dict.fromkeys(vlm_qs or []))
    except Exception:
        auto_queries = []

    # 4) 检索规范库（按 Query 聚合）
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for q in auto_queries:
        try:
            res = kb.search(q, top_k=5)
            all_results[q] = res
        except Exception:
            all_results[q] = []

    kb_snippets = format_kb_context(all_results, max_items=12)

    # 5) 若 VLM 支持单次整体合成 → 直接调用
    if hasattr(vlm, "render_final"):
        try:
            print("render_final start")
            out = vlm.render_final(
                image=image,
                scene_guess=scene_guess,
                auto_queries=auto_queries,
                kb_snippets=kb_snippets,
                schema_contract=SCHEMA_TEXT,
            )
            _repair_output(out)
            return out
        except Exception:
            pass

    # 6) 程序化回退：把 proposals + kb_snippets 组装为最终 JSON
    hazards = []
    for i, pr in enumerate(proposals or [HazardProposal("综合", "未能明确识别隐患（需补拍）", [0,0,0,0], "low_res", 0.25)], start=1):
        severity, (like, expo, consq) = _estimate_severity_and_factors(pr.label, pr.confidence)
        risk = _score_risk(like, expo, consq)
        citations = []
        # 从全体 snippets 中取 1-2 条，真实生产中应按 hazard 相关度再筛
        for s in kb_snippets[:2]:
            citations.append({
                "doc_id": s.get("doc_id", ""),
                "clause_id": s.get("clause_id", ""),
                "clause_brief": (s.get("clause_text", "")[:40] or "未检出可直接匹配的条款"),
                "match_type": "direct" if s else "NO_KB_MATCH",
            })
        if not citations:
            citations = [{"doc_id": "", "clause_id": "", "clause_brief": "未检出可直接匹配的条款", "match_type": "NO_KB_MATCH"}]
        hazards.append({
            "id": f"HZ-{i:04d}",
            "category": pr.category,
            "label": pr.label,
            "evidence_region": {"type": "bbox", "bbox_xywh_norm": pr.bbox_xywh_norm} if pr.bbox_xywh_norm != [0,0,0,0] else {"type": "global"},
            "evidence_quality": pr.evidence_quality,
            "kb_citations": citations,
            "severity": severity,
            "risk_score": risk,
            "confidence": round(max(0.0, min(1.0, pr.confidence)), 3),
            "rationale_brief": ("画面存在疑似违规点，需按规范整改" if pr.confidence < 0.5 else "画面可见明确违规现象"),
            "remediation": [
                "设置临时隔离/警戒并停止相关作业" if "临边" in pr.label else "立即清理/隔离风险点并张贴警示",
                "按规范配置永久性防护与点检制度" if "临边" in pr.label else "建立巡检清单与培训并量化考核",
            ],
            "status": "CONFIRMED" if pr.confidence >= 0.5 else "NEEDS_REVIEW",
        })

    out = {
        "task_id": task_id or str(uuid.uuid4()),
        "scene_guess": scene_guess,
        "auto_queries": auto_queries,
        "overall_risk_score": max([h["risk_score"] for h in hazards]) if hazards else 0,
        "hazards": hazards,
        "missing_info": (["补拍：全景+中景+近景（含标识/尺度尺），提高分辨率与光照"] if any(h.get("status") == "NEEDS_REVIEW" or h.get("evidence_quality") in {"blurry","occluded","low_res"} for h in hazards) else []),
        "version": "img-only-v1.0",
    }
    _repair_output(out)
    return out

# -----------------------------
# Output repair & summary
# -----------------------------

def _repair_output(out_json: Dict[str, Any]) -> None:
    try:
        for h in out_json.get("hazards", []):
            reg = h.get("evidence_region", {})
            if isinstance(reg, dict) and reg.get("type") == "bbox":
                bb = reg.get("bbox_xywh_norm", [])
                if isinstance(bb, list) and len(bb) == 4:
                    reg["bbox_xywh_norm"] = [float(max(0.0, min(1.0, float(v)))) for v in bb]
        scores = [int(h.get("risk_score", 0)) for h in out_json.get("hazards", [])]
        out_json["overall_risk_score"] = int(max(scores) if scores else 0)
        out_json.setdefault("version", "img-only-v1.0")
        out_json.setdefault("task_id", str(uuid.uuid4()))
    except Exception:
        pass

def build_summary(out_json: Dict[str, Any]) -> str:
    hz = out_json.get("hazards", [])
    n = len(hz)
    top = max((h.get("risk_score", 0) for h in hz), default=0)
    scene = out_json.get("scene_guess", {}).get("scene", "")
    return f"场景：{scene}。检测到隐患{n}项，最高风险分{top}。已给出条款引用与整改建议。"


def mrag_run(image_path:str):
    hazard_cfg = "D:/LLM/project/ustarconstruct/backend/config/safety_config.yaml"
    p = pathlib.Path(hazard_cfg)
    if p.suffix.lower() in {".json"}:
        hazard_cfg = HazardConfig.from_dict(json.loads(pathlib.Path(p).read_text("utf-8")))
    elif p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            print("PyYAML 未安装，无法读取 YAML 配置", file=sys.stderr)
            sys.exit(2)
        hazard_cfg = yaml.safe_load(pathlib.Path(p).read_text("utf-8"))

    kb: KBClient = KBClient()  # replace with your implementation
    vlm = OpenAIVLM() 
    result = analyze_image(image_path, kb=kb, hazard_cfg=hazard_cfg, vlm=vlm)
    result['image_path'] = image_path
    with open("output.json","w",encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(build_summary(result))
    return result
    

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # import argparse, sys, pathlib

    # parser = argparse.ArgumentParser(description="Image-only Safety Compliance RAG (V2)")
    # parser.add_argument("image", nargs="?", default=None, help="Path to image file")
    # parser.add_argument("--hazard-cfg", dest="hazard_cfg", default=None, help="Path to hazard config (json/yaml)")
    # parser.add_argument("--no-dummy-kb", action="store_true", help="Disable dummy KB and use your own KB client")
    # parser.add_argument("--use-dummy-vlm", default="vlm",action="store_true", help="Force DummyVLM (for testing)")
    # args = parser.parse_args()

    # load hazard cfg
    hazard_cfg = "D:/LLM/project/ustarconstruct/backend/config/safety_config.yaml"
    p = pathlib.Path(hazard_cfg)
    if p.suffix.lower() in {".json"}:
        hazard_cfg = HazardConfig.from_dict(json.loads(pathlib.Path(p).read_text("utf-8")))
    elif p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            print("PyYAML 未安装，无法读取 YAML 配置", file=sys.stderr)
            sys.exit(2)
        hazard_cfg = yaml.safe_load(pathlib.Path(p).read_text("utf-8"))

    kb: KBClient = KBClient()  # replace with your implementation
    vlm = OpenAIVLM() 
    test_image = "wechat_2025-10-22_144349_916.png"
    result = analyze_image(test_image, kb=kb, hazard_cfg=hazard_cfg, vlm=vlm)
    result['image_path'] = test_image
    with open("output.json","w",encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(build_summary(result))
