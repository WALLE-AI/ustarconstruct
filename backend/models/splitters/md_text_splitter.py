import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

TITLE_RE = re.compile(r'^\s*#{1,6}\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$')        # 匹配形如 "# 2.3 安全行为要求"
SUBNUM_HEADING_RE = re.compile(r'^\s*#{1,6}\s*(\d+\.\d+\.\d+)\s+(.+?)\s*$') # 匹配形如 "# 2.3.1 建设单位"
SUBNUM_INLINE_RE  = re.compile(r'^\s*(\d+\.\d+\.\d+)\s+(.+?)\s*$')          # 匹配形如 "2.1.16 鼓励……"

def chunk_md_by_numbered_subtitles(md_text: str,file_name:str) -> List[Dict[str, str]]:
    """
    将 Markdown 文本分块：
    - title：采用“二级编号标题”（形如 2.3）的最近父级标题
    - sub_title：采用“三级编号”（形如 2.3.1 / 2.1.16），可以是带 # 的标题行，或普通正文行起始编号
    - sub_text：该三级编号直到下一个三级编号/新的二级编号/文末之间的内容（含多行）
    """
    lines = md_text.splitlines()

    chunks: List[Dict[str, str]] = []
    current_title_num: Optional[str] = None
    current_title_text: Optional[str] = None

    current_sub_num: Optional[str] = None
    current_sub_text: List[str] = []
    current_sub_title_text: Optional[str] = None

    def flush_sub():
        nonlocal current_sub_num, current_sub_title_text, current_sub_text
        if current_title_text and current_sub_num:
            sub_text = "\n".join(current_sub_text).rstrip()
            data = {
                "title": f"{current_title_num} {current_title_text}".strip(),
                "sub_title": f"{current_sub_num} {current_sub_title_text}".strip(),
                "sub_text": sub_text
            }
            data["file_name"] = file_name
            chunks.append(
                Document(
                    page_content=json.dumps(data,ensure_ascii=False),
                    metadata=data
                )
            )
        current_sub_num = None
        current_sub_title_text = None
        current_sub_text = []

    for raw in lines:
        line = raw.rstrip()

        # 1) 先识别是否是带 # 的标题行
        m_heading = TITLE_RE.match(line)
        if m_heading:
            number_str, title_text = m_heading.group(1), m_heading.group(2)

            # 如果是三级编号（2.3.1 …）——它既改变 sub_title，又可能改变父 title
            m_sub_h = SUBNUM_HEADING_RE.match(line)
            if m_sub_h:
                # 刷新之前 sub_title
                flush_sub()
                subnum, subtext = m_sub_h.group(1), m_sub_h.group(2)

                # 父 title 采用其“上一级编号”（例如 2.3）
                parent_num = ".".join(subnum.split(".")[:2])
                # 如果当前 title 不等于这个 parent，就更新
                if current_title_num != parent_num:
                    current_title_num = parent_num
                    # 父标题文本：若恰好这一行也出现其父标题，会在后续遇到；这里先用 m_heading 的 title_text 兜底
                    # 但更稳妥：当遇到明确的 "2.3 ..." 标题时再覆盖。此处尝试保留已有父标题文本。
                    if current_title_text is None or current_title_num not in (current_title_num or ""):
                        # 暂存占位，若后续遇到明确“2.3 …”再覆盖
                        current_title_text = f"(章节 {current_title_num})"

                # 开始新的 sub
                current_sub_num = subnum
                current_sub_title_text = subtext
                continue

            # 否则是一般编号标题（可能是“2 …”、“2.3 …”等）
            # 如果是“二级编号”（一个点），把它作为父 title
            if number_str.count(".") == 1:
                # 刷新正在累积的 sub（因为父标题切换了）
                flush_sub()
                current_title_num = number_str
                current_title_text = title_text
                continue

            # 其他层级的编号标题（如 "2" 或 "2.3.4.5"），只当作结构提示，必要时刷新 sub
            if number_str.count(".") != 1:
                # 不改变父 title，但如已在收集 sub_text，遇到更高层也应结束该 sub
                # 这里仅当遇到顶层（无点）或更高层时刷新
                if current_sub_num:
                    flush_sub()
                continue

        # 2) 如果不是 # 标题，再识别是否是行内的三级编号起始（如 "2.1.16 …"）
        m_inline = SUBNUM_INLINE_RE.match(line)
        if m_inline and m_inline.group(1).count(".") == 2:
            # 注意：这可能出现在段落中（你的文档 2.1.1~2.1.16 多为这种形式）
            # 新 sub 开始，先刷新旧的
            flush_sub()
            current_sub_num = m_inline.group(1)
            current_sub_title_text = m_inline.group(2)

            # 父 title 为其前两段编号
            parent_num = ".".join(current_sub_num.split(".")[:2])
            if current_title_num != parent_num:
                current_title_num = parent_num
                if current_title_text is None:
                    current_title_text = f"(章节 {current_title_num})"
            continue

        # 3) 普通正文行：若已在某个 sub 内，累积；否则忽略
        if current_sub_num:
            current_sub_text.append(line)

    # 文件结束，收尾
    flush_sub()
    return chunks


def save_jsonl(chunks: List[Dict[str, str]], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 把路径换成你的实际文件；我已填入你上传的文件名
    MD_PATH = Path("data/test_markdown/住宅工程质量常见问题防治技术规程( DB341659-2022)/vlm/住宅工程质量常见问题防治技术规程( DB341659-2022).md")
    text = MD_PATH.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_md_by_numbered_subtitles(text,Path(MD_PATH).name)

    # 示例：打印若干条，或筛选出 2.3.1
    # for c in chunks[:5]:
    #     print(c["title"], "|", c["sub_title"])
    # print("\n--- Filter 2.3.1 ---")
    # for c in chunks:
    #     if c["sub_title"].startswith("2.3.1 "):
    #         print(json.dumps(c, ensure_ascii=False, indent=2)[:800])  # 预览前 800 字
    #         break

    # 可选：保存为 JSONL
    save_jsonl(chunks, Path("output.jsonl"))
