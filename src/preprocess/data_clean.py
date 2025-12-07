import argparse
import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple, Union

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


def extract_body_for_dedup(text: str) -> str:
    """提取正文用于去重，去除标题行（通常是第一行，包含分局名称等）。
    
    这样可以让同一份文件在不同分局网站的副本被识别为重复。
    """
    if not text:
        return ""
    # 按换行或中文句号分割，跳过第一段（通常是标题）
    # 先尝试按换行分割
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) > 1:
        # 跳过第一行（标题行）
        return ' '.join(lines[1:])
    # 如果没有换行，尝试按第一个句号分割
    parts = text.split('。', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return text


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """提取字符级 n-gram 集合（适合中文）。"""
    text = re.sub(r'\s+', '', text)  # 去除空白
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def get_char_ngram_counter(text: str, n: int = 3) -> Counter:
    """提取字符级 n-gram 计数器（用于加权相似度）。"""
    text = re.sub(r'\s+', '', text)
    if len(text) < n:
        return Counter([text]) if text else Counter()
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """计算 Jaccard 相似度。"""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def containment_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """计算包含度相似度：交集占较小集合的比例。
    
    用于检测一个文本是否是另一个的子集/超集。
    """
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return inter / min_size if min_size > 0 else 0.0


def get_key_phrases(text: str, phrase_len: int = 8) -> Set[str]:
    """提取关键短语（连续字符片段），用于检测模板化内容。
    
    年报、政府文件等常包含固定的模板表述，通过检测这些关键短语的重叠可以识别模板化重复。
    """
    # 去除空白、数字和标点，保留核心文字
    text = re.sub(r'[\s\d\.\,;:!?，。；：！？、\"\"''（）\[\]【】\n\r]', '', str(text))
    if len(text) < phrase_len:
        return {text} if text else set()
    return {text[i:i+phrase_len] for i in range(len(text) - phrase_len + 1)}


def key_phrase_overlap(phrases_a: Set[str], phrases_b: Set[str], 
                       min_overlap: int = 10) -> bool:
    """检测两个文档是否有足够多的公共关键短语。
    
    用于检测模板化内容（如年报中的固定表述、政府文件模板等）。
    """
    if not phrases_a or not phrases_b:
        return False
    common = phrases_a & phrases_b
    return len(common) >= min_overlap


def is_too_similar_text(a: str, b: str,
                        jaccard_threshold: float = 0.5,
                        containment_threshold: float = 0.8,
                        phrase_overlap_threshold: int = 10,
                        ngram_size: int = 3,
                        phrase_len: int = 8) -> bool:
    """基于字符级 n-gram 和关键短语的中文文本相似度检测。

    使用三种策略：
    1. Jaccard 相似度 ≥ jaccard_threshold 视为重复
    2. 包含度相似度 ≥ containment_threshold 视为重复（处理子集/超集关系）
    3. 关键短语重叠 ≥ phrase_overlap_threshold 视为模板化重复
    """
    if not a or not b:
        return False
    
    # 长度差异过大时快速跳过
    len_a, len_b = len(a), len(b)
    if len_a > 0 and len_b > 0:
        ratio = min(len_a, len_b) / max(len_a, len_b)
        if ratio < 0.3:  # 长度差异超过 3 倍，不太可能是重复
            return False
    
    ngrams_a = get_char_ngrams(a, ngram_size)
    ngrams_b = get_char_ngrams(b, ngram_size)
    
    # Jaccard 相似度
    jacc = jaccard_similarity(ngrams_a, ngrams_b)
    if jacc >= jaccard_threshold:
        return True
    
    # 包含度相似度（检测子集关系）
    cont = containment_similarity(ngrams_a, ngrams_b)
    if cont >= containment_threshold:
        return True
    
    # 关键短语重叠（检测模板化内容）
    phrases_a = get_key_phrases(a, phrase_len)
    phrases_b = get_key_phrases(b, phrase_len)
    if key_phrase_overlap(phrases_a, phrases_b, phrase_overlap_threshold):
        return True
    
    return False


def make_bucket_key(norm_text: str, ngram_size: int = 5, top_k: int = 10) -> str:
    """为近似去重构造一个粗粒度 bucket key（基于高频 n-gram）。

    - 提取字符级 n-gram 并统计频率
    - 取频率最高的 top_k 个 n-gram
    - 排序后连接成桶键
    
    具有相似高频 n-gram 的文本会落在相同桶中。
    """
    text = re.sub(r'\s+', '', norm_text)
    if len(text) < ngram_size:
        return text if text else ""
    
    ngram_counter = get_char_ngram_counter(text, ngram_size)
    # 取频率最高的 top_k 个
    top_ngrams = [ng for ng, _ in ngram_counter.most_common(top_k)]
    if not top_ngrams:
        return ""
    return "|".join(sorted(top_ngrams))


def compute_content_hash(text: str) -> str:
    """计算内容的哈希指纹，用于快速精确去重。
    
    对正文部分（去除标题）计算哈希，这样同一份文件在不同分局的副本会有相同哈希。
    """
    body = extract_body_for_dedup(text)
    norm = normalize_text(body)
    # 只取正文的核心部分（前500字符）计算哈希，避免尾部差异
    core = norm[:500] if len(norm) > 500 else norm
    return hashlib.md5(core.encode('utf-8')).hexdigest()


def compute_simhash(text: str, ngram_size: int = 3, hash_bits: int = 64) -> int:
    """计算 SimHash 指纹，用于快速近似去重。
    
    SimHash 是局部敏感哈希，相似文本的 SimHash 值汉明距离较小。
    """
    text = re.sub(r'\s+', '', text)
    if not text:
        return 0
    
    ngrams = get_char_ngram_counter(text, ngram_size)
    if not ngrams:
        return 0
    
    # 初始化 hash_bits 维向量
    v = [0] * hash_bits
    
    for ngram, weight in ngrams.items():
        # 对每个 n-gram 计算哈希
        h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += weight
            else:
                v[i] -= weight
    
    # 生成最终指纹
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """计算两个整数的汉明距离。"""
    return bin(h1 ^ h2).count('1')


def is_simhash_similar(h1: int, h2: int, threshold: int = 10) -> bool:
    """判断两个 SimHash 是否相似（汉明距离 ≤ threshold）。"""
    return hamming_distance(h1, h2) <= threshold


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

    # 不做 chunk 切分，直接返回整条清洗后的文本
    new_doc: Dict[str, Union[str, Dict]] = {
        "id": rid,
        "contents": text,
    }
    meta = record.get("metadata")
    if isinstance(meta, dict):
        new_doc["metadata"] = dict(meta)

    return [new_doc]


def process_file(input_path: str, output_path: str, 
                 simhash_threshold: int = 8,
                 jaccard_threshold: float = 0.5,
                 containment_threshold: float = 0.8,
                 phrase_overlap_threshold: int = 10,
                 phrase_len: int = 8) -> None:
    """处理文件，进行多层去重。
    
    去重策略（五层）：
    1. 精确哈希去重：基于正文前500字符的 MD5
    2. 精确文本去重：完全相同的标准化文本
    3. SimHash 近似去重：汉明距离 ≤ simhash_threshold 视为重复
    4. N-gram Jaccard/包含度去重：在同一 bucket 内精细比较
    5. 关键短语重叠去重：检测模板化内容（年报、政府文件等）
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        simhash_threshold: SimHash 汉明距离阈值，越小越严格（默认 8）
        jaccard_threshold: Jaccard 相似度阈值（默认 0.5）
        containment_threshold: 包含度相似度阈值（默认 0.8）
        phrase_overlap_threshold: 关键短语重叠数量阈值（默认 10）
        phrase_len: 关键短语长度（默认 8）
    """
    cleaned: List[Dict] = []
    seen_texts: Set[str] = set()
    seen_hashes: Set[str] = set()  # 基于正文哈希的快速去重
    simhash_index: Dict[int, List[Tuple[int, str]]] = {}  # SimHash 索引：bucket -> [(simhash, text)]
    buckets: Dict[str, List[Tuple[str, Set[str]]]] = {}  # N-gram bucket 索引：bucket -> [(text, phrases)]
    
    dup_stats = {"hash": 0, "exact": 0, "simhash": 0, "ngram": 0, "phrase": 0}

    for rec in tqdm(load_jsonl(input_path), desc="清洗与去重", unit="doc"):
        docs = clean_record(rec)
        for d in docs:
            contents = d.get("contents")
            # 仅对字符串内容做去重；结构化 dict 保留
            if isinstance(contents, str):
                norm = normalize_text(contents)
                if not norm:
                    continue
                
                # 第一层：基于正文哈希的快速去重
                content_hash = compute_content_hash(contents)
                if content_hash in seen_hashes:
                    dup_stats["hash"] += 1
                    continue
                
                # 第二层：精确文本去重
                if norm in seen_texts:
                    dup_stats["exact"] += 1
                    continue
                
                # 提取正文（去除标题）用于近似去重
                body_norm = normalize_text(extract_body_for_dedup(contents))
                text_for_dedup = body_norm if body_norm else norm
                
                # 第三层：SimHash 近似去重（O(1) 查找）
                simhash = compute_simhash(text_for_dedup)
                # 使用 SimHash 的高位作为 bucket key，减少比较次数
                simhash_bucket = simhash >> 48  # 取高 16 位作为 bucket
                is_simhash_dup = False
                if simhash_bucket in simhash_index:
                    for stored_hash, _ in simhash_index[simhash_bucket]:
                        if is_simhash_similar(simhash, stored_hash, simhash_threshold):
                            is_simhash_dup = True
                            break
                if is_simhash_dup:
                    dup_stats["simhash"] += 1
                    continue
                
                # 预计算关键短语（用于第四、五层）
                phrases = get_key_phrases(text_for_dedup, phrase_len)
                
                # 第四层：N-gram Jaccard/包含度去重（精细比较）
                bucket_key = make_bucket_key(text_for_dedup)
                is_dup = False
                dup_type = None
                
                if bucket_key:
                    bucket_list = buckets.setdefault(bucket_key, [])
                    for kept_text, kept_phrases in bucket_list:
                        # 检查 Jaccard/包含度相似度
                        ngrams_a = get_char_ngrams(text_for_dedup)
                        ngrams_b = get_char_ngrams(kept_text)
                        
                        jacc = jaccard_similarity(ngrams_a, ngrams_b)
                        if jacc >= jaccard_threshold:
                            is_dup = True
                            dup_type = "ngram"
                            break
                        
                        cont = containment_similarity(ngrams_a, ngrams_b)
                        if cont >= containment_threshold:
                            is_dup = True
                            dup_type = "ngram"
                            break
                        
                        # 第五层：关键短语重叠去重（检测模板化内容）
                        if key_phrase_overlap(phrases, kept_phrases, phrase_overlap_threshold):
                            is_dup = True
                            dup_type = "phrase"
                            break
                
                if is_dup:
                    dup_stats[dup_type] += 1
                    continue
                
                # 添加到 bucket
                if bucket_key:
                    buckets[bucket_key].append((text_for_dedup, phrases))

                # 通过所有去重检查，保留该文档
                seen_hashes.add(content_hash)
                seen_texts.add(norm)
                simhash_index.setdefault(simhash_bucket, []).append((simhash, text_for_dedup))
                
            cleaned.append(d)
    
    # 输出去重统计
    total_dup = sum(dup_stats.values())
    print(f"\n去重统计:")
    print(f"  - 哈希去重: {dup_stats['hash']}")
    print(f"  - 精确去重: {dup_stats['exact']}")
    print(f"  - SimHash 去重: {dup_stats['simhash']}")
    print(f"  - N-gram Jaccard/包含度去重: {dup_stats['ngram']}")
    print(f"  - 关键短语重叠去重: {dup_stats['phrase']}")
    print(f"  - 总去重数: {total_dup}")
    print(f"  - 保留文档数: {len(cleaned)}")
    
    save_jsonl(output_path, cleaned)


def main():
    parser = argparse.ArgumentParser(description="清洗与去重文本数据（支持五层去重策略）")
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
    parser.add_argument(
        "--simhash_threshold",
        type=int,
        default=8,
        help="SimHash 汉明距离阈值，越小越严格（默认 8）",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.5,
        help="Jaccard 相似度阈值（默认 0.5）",
    )
    parser.add_argument(
        "--containment_threshold",
        type=float,
        default=0.8,
        help="包含度相似度阈值（默认 0.8）",
    )
    parser.add_argument(
        "--phrase_overlap",
        type=int,
        default=10,
        help="关键短语重叠数量阈值，用于检测模板化内容（默认 10）",
    )
    parser.add_argument(
        "--phrase_len",
        type=int,
        default=8,
        help="关键短语长度（默认 8）",
    )
    args = parser.parse_args()

    process_file(
        args.input, 
        args.output,
        simhash_threshold=args.simhash_threshold,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
        phrase_overlap_threshold=args.phrase_overlap,
        phrase_len=args.phrase_len
    )

def read_jsonl(file_path: str, line_min: int = 10, line_max: int = 100) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    main()

    # res = read_jsonl("datasets/OmniEval-Corpus/all_data_clean.jsonl")
    # print(len(res))
    # print(res[10:20])

