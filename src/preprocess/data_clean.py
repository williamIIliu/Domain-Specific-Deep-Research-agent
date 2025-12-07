import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Union, Set

from tqdm import tqdm


def load_jsonl(path: str) -> Iterable[Dict]:
    """按行读取 JSONL，解析失败的行跳过。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def save_jsonl(path: str, docs: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """用于精确去重的文本标准化：去首尾空白、多余空白统一为单空格。"""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_too_similar_text(a: str, b: str,
                        word_threshold: float = 0.9,
                        bigram_threshold: float = 0.7) -> bool:
    """用于去重的全局近似相似度：

    - 词级 Jaccard ≥ word_threshold 视为重复；
    - 或 bigram Jaccard ≥ bigram_threshold 视为重复。"""
    if not a or not b:
        return False

    tokens_a_list = str(a).split()
    tokens_b_list = str(b).split()
    if not tokens_a_list or not tokens_b_list:
        return False

    tokens_a = set(tokens_a_list)
    tokens_b = set(tokens_b_list)

    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    word_j = inter / union if union > 0 else 0.0
    if word_j >= word_threshold:
        return True

    def build_bigrams(tokens):
        return {f"{tokens[i]}|||{tokens[i+1]}" for i in range(len(tokens) - 1)}

    bigrams_a = build_bigrams(tokens_a_list)
    bigrams_b = build_bigrams(tokens_b_list)
    if not bigrams_a or not bigrams_b:
        return False
    inter_bg = len(bigrams_a & bigrams_b)
    union_bg = len(bigrams_a | bigrams_b)
    bigram_j = inter_bg / union_bg if union_bg > 0 else 0.0
    return bigram_j >= bigram_threshold


def make_bucket_key(norm_text: str, max_tokens: int = 20) -> str:
    """为近似去重构造一个粗粒度 bucket key。

    - 按空白切分 token；
    - 去重后排序，截断到前 max_tokens 个；
    - 连接成一个字符串作为桶键。
    具有相似词袋的文本会落在相同桶中，从而只在桶内做精细相似度比较。"""
    tokens = norm_text.split()
    if not tokens:
        return ""
    uniq_sorted = sorted(set(tokens))[:max_tokens]
    return " ".join(uniq_sorted)


FOOTER_PATTERNS = [
    # 各地分局网站页脚 + ICP + 公安备案号
    r"法律声明\s*\|[\s\S]*?京icp备[0-9a-zA-Z]+号[\s\S]*?京公网安备[0-9]+号",
    r"国家外汇管理局[\S ]+分局主办[\S ]*?网站标识码bm[0-9]+",
    r"网站标识码bm[0-9]+",
]

LINE_DROP_KEYWORDS = [
    "举报本回复",
    "模拟交易:模拟炒股免费实操交易技能",
    "微牛证券",
    "webull",
    "版权所有",
]


def strip_node_prefix(text: str) -> str:
    """去掉 pkl 抽取时生成的 'node id: ... text:' 前缀。"""
    return re.sub(r"^node id:\s*\S+\s+text:\s*", "", text.strip(), flags=re.IGNORECASE)


def remove_noise(text: str) -> str:
    """移除网页页脚、明显广告/导航等噪声。"""
    # 页脚类整体模式
    for pat in FOOTER_PATTERNS:
        text = re.sub(pat, "", text)

    # 行级噪声：按换行切分，过滤含关键词的行
    parts = re.split(r"[\n\r]", text)
    cleaned_lines: List[str] = []
    for line in parts:
        if any(k in line for k in LINE_DROP_KEYWORDS):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return text.strip()


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cn / total if total > 0 else 0.0


def _split_paragraphs(text: str) -> List[str]:
    """按段落拆分：优先用空行/多换行，其次单换行。"""
    # 先统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 以两个及以上换行视为段落边界
    paras = re.split(r"\n{2,}", text)
    refined: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # 对于内部仍然存在的大量换行，简单合并
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        refined.append("".join(lines))
    return refined


def _split_sentences(text: str) -> List[str]:
    """备用：按句号/问号/感叹号/分号拆句。"""
    sentences: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？!?；;":
            sent = "".join(buf).strip()
            if sent:
                sentences.append(sent)
            buf = []
    if buf:
        tail = "".join(buf).strip()
        if tail:
            sentences.append(tail)
    return sentences


def split_into_chunks(text: str,
                      min_len: int = 200,
                      max_len: int = 800) -> List[str]:
    """纯句子级 chunk 切分。

    - 对全文调用 _split_sentences 得到句子序列；
    - 按原始顺序累加句子，直到接近 max_len，超过则起新 chunk；
    - 不为了凑够 min_len 强行跨块合并，长度 < min_len 的 chunk 仍保留；
    - 仅当单句 > max_len 时，对该句做硬切。"""

    if not text:
        return []

    # 先按句子拆分整个文本
    sentences = _split_sentences(text)
    if not sentences:
        # 退化情况：没有句号等分隔符时，按长度硬切
        norm = re.sub(r"\s+", " ", text).strip()
        if not norm:
            return []
        chunks = [norm[i:i + max_len].strip() for i in range(0, len(norm), max_len)]
        return [c for c in chunks if chinese_ratio(c) >= 0.2]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        s_len = len(s)

        # 单句本身超过 max_len：先把已有缓存收尾，再对该句硬切
        if s_len > max_len:
            if cur:
                chunks.append("".join(cur).strip())
                cur = []
                cur_len = 0
            for i in range(0, s_len, max_len):
                piece = s[i:i + max_len].strip()
                if piece:
                    chunks.append(piece)
            continue

        if cur_len + s_len <= max_len:
            cur.append(s)
            cur_len += s_len
        else:
            if cur:
                chunks.append("".join(cur).strip())
            cur = [s]
            cur_len = s_len

    if cur:
        chunks.append("".join(cur).strip())

    # 只保留中文比例足够的 chunk；不再因为 < min_len 而删除
    return [c for c in chunks if chinese_ratio(c) >= 0.2]


def clean_record(record: Dict) -> List[Dict]:
    """对单条样本做清洗与拆分，返回 0~N 条新样本。"""
    rid = record.get("id")
    contents = record.get("contents")

    # 结构化数据：保持一条原样返回
    if isinstance(contents, dict):
        return [record]

    # 文本数据：清洗 & 拆分
    text = str(contents or "").strip()
    if not text:
        return []

    text = strip_node_prefix(text)
    text = remove_noise(text)

    # 专门过滤 "只有标题+页脚" 的 GOV 短文本：既包含页脚关键词，又整体很短
    if (
        len(text) < 200
        and "法律声明 | 联系我们" in text
        and "京icp备" in text
    ):
        return []

    # 过滤极短或几乎没有中文的文本
    if not text or len(text) < 50 or chinese_ratio(text) < 0.1:
        return []

    chunks = split_into_chunks(text)
    if not chunks:
        return []

    cleaned_docs: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        new_doc: Dict[str, Union[str, Dict]] = {
            "id": f"{rid}#{idx+1}",
            "contents": chunk,
        }
        meta = record.get("metadata")
        if isinstance(meta, dict):
            new_meta = dict(meta)
            new_meta["source_id"] = rid
            new_meta["chunk_index"] = idx + 1
            new_doc["metadata"] = new_meta
        cleaned_docs.append(new_doc)

    return cleaned_docs


def process_file(input_path: str, output_path: str) -> None:
    cleaned: List[Dict] = []
    seen_texts: Set[str] = set()
    buckets: Dict[str, List[str]] = {}

    for rec in tqdm(load_jsonl(input_path), desc="清洗与拆分", unit="doc"):
        docs = clean_record(rec)
        for d in docs:
            contents = d.get("contents")
            # 仅对字符串内容做精确去重；结构化 dict 保留
            if isinstance(contents, str):
                norm = normalize_text(contents)
                if not norm:
                    continue
                if norm in seen_texts:
                    continue
                # 近似去重：只在相同 bucket 内比较，降低复杂度
                bucket_key = make_bucket_key(norm)
                if bucket_key:
                    bucket_list = buckets.setdefault(bucket_key, [])
                    is_dup = False
                    for kept in bucket_list:
                        if is_too_similar_text(norm, kept):
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    bucket_list.append(norm)

                seen_texts.add(norm)
            cleaned.append(d)
    save_jsonl(output_path, cleaned)


def main():
    parser = argparse.ArgumentParser(description="清洗与拆分 OmniEval-Corpus 文本数据")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/OmniEval-Corpus/all_data_raw.jsonl",
        help="原始 JSONL 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/OmniEval-Corpus/all_data_clean.jsonl",
        help="清洗后的 JSONL 输出路径",
    )
    args = parser.parse_args()

    process_file(args.input, args.output)

def read_jsonl(file_path: str, line_min: int = 10, line_max: int = 100) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    # main()

    res = read_jsonl("datasets/OmniEval-Corpus/all_data_clean.jsonl")
    print(len(res))
    print(res[10:20])

