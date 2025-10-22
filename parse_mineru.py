import base64
import os
from pathlib import Path
import re
from typing import Dict,Any, List
import json
from urllib.parse import unquote
from loguru import logger
import requests

PARSE_TIMEOUT = 1800    # nginx配置30分钟超时
MINERU_CFG = {
    # 'output_dir': './outputs',
    # 'lang_list': 'ch',    # [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|th|el|latin|arabic|east_slavic|cyrillic|devanagari]，文档语言类型，可提升 OCR 准确率，仅用于 pipeline 后端
    # 'backend': 'vlm-sglang-engine', # [pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client] 解析后端（默认为 pipeline）
    # 'parse_method': 'auto', # [auto|txt|ocr] 解析方法
    # 'formula_enable': 'true', # 是否解析公式，默认开启
    # 'table_enable': 'true', # 是否解析表格，默认开启
    # 'server_url': '', 服务器url,使用client模式时的服务器参数
    # 'return_md': 'true', 是否返回markdown格式文档，默认开启
    'return_middle_json': 'true',   # 返回中间结果，默认关闭
    # 'return_model_output': 'true',    # 返回模型输出，仅用于 pipeline 后端，默认关闭
    # 'return_content_list': 'true',    # 返回简要结果（没有坐标框等详细信息），默认关闭
    'return_images': 'true',  # 是否以base64形式返回模型图片，默认关闭
    # 'start_page_id': 0,   # 起始页，默认0
    # 'end_page_id': 99999, # 结束页，默认99999
    'extract_catalog': 'false' # 是否解析目录，默认开启
}
PARSE_URL = 'http://36.103.251.31:9005/mineru/file_parse'

def parse_pdf(file_path: Path, file_name: Path) -> Dict[str, Any]:
    """解析PDF文件"""
    logger.info(f'开始解析: {file_name}')
    
    with open(file_path, 'rb') as f:
        files = {'files': (str(file_name), f, 'application/pdf')}
        data = MINERU_CFG
        response = requests.post(
            PARSE_URL, 
            files=files, 
            data=data, 
            timeout=PARSE_TIMEOUT
        )
        
    response.raise_for_status()
    res = response.json()
    file_name_dir = 'data/markdown/'+file_name.split('.pdf')[0]
    if not os.path.exists(file_name_dir):
        os.makedirs(file_name_dir)
    for key,value in res['results'].items():  
        md_content = value.get('md_content', '')
        middle_json = value.get('middle_json', '')
        md_content_path = os.path.join(file_name_dir, key+".md")
        with open(md_content_path,"w",encoding="utf-8") as file:
            file.write(md_content)
        middle_json_path = os.path.join(file_name_dir, key+".jsonl")
        with open(middle_json_path,"w",encoding="utf-8") as file:
            file.write(middle_json)
        logger.info(f'Markdown内容已提取, 长度: {len(md_content)} 字符')
        images = value.get('images', {})
        for img_name, img_data in images.items():
            if not os.path.exists(file_name_dir+'/images'):
                os.makedirs(file_name_dir+'/images')
            try:
                m = re.match(r"^data:(image/[\w\+\-\.]+);base64,(.*)$", img_data, flags=re.S | re.I)
                mime = None
                data = img_data
                if m:
                    mime, data = m.group(1).lower(), m.group(2)

                # 2) 清洗+补 padding
                data = re.sub(r"\s+", "", data)
                if (pad := (-len(data)) % 4):
                    data += "=" * pad
                raw = base64.b64decode(data, validate=False)
            except Exception:
            # 兼容 url-safe base64
                raw = base64.urlsafe_b64decode(data)
                
                # 依据 mime 猜扩展名
                ext_map = {
                    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
                    "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp",
                    "image/tiff": "tiff", "image/x-icon": "ico", "image/svg+xml": "svg",
                }
                ext = "." + ext_map.get(mime, "bin")
            img_path = os.path.join(file_name_dir+'/images', img_name)
            with open(img_path, 'wb') as img_file:
                img_file.write(raw)
            logger.info(f'图片已保存: {img_path}')
    return md_content


def list_pdfs_friendly(
    dir_path: str,
    recursive: bool = True,
    sort_result: bool = True
) -> List[Dict[str, str]]:
    """
    返回包含两列的字典数组：
      - path: 可读中文的绝对路径（适合人类阅读/日志）
      - url : 可点击的 file:// URL（可能含%编码，浏览器/Markdown可用）
    """
    base = Path(dir_path).expanduser().resolve(strict=False)
    if not base.exists() or not base.is_dir():
        return []

    it = base.rglob("*.pdf") if recursive else base.glob("*.pdf")

    rows: List[Dict[str, str]] = []
    for p in it:
        if p.is_file():
            try:
                p_abs = p.resolve(strict=True)
                url = p_abs.as_uri()           # 规范URL（含百分号编码）
                rows.append({"path": str(p_abs), "url": url, "url_readable": unquote(url)})
            except FileNotFoundError:
                continue

    if sort_result:
        rows.sort(key=lambda d: d["path"].lower())
    return rows
    
    
if __name__ == '__main__':
    pdf_links = list_pdfs_friendly("data/pdf", recursive=True)
    for  file_link in pdf_links[1:]:
        file_path = file_link['path']
        file_name = Path(file_path).name
        parse_pdf(file_path, file_name)