import argparse
import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple, Union
import logging
from tqdm import tqdm


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> Iterable[Dict]:
    """按行读取 JSONL，解析失败的行跳过。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")
    
    total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8'))
    failed_count = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                failed_count += 1
                if failed_count <= 10:  # 只打印前10个错误
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")
                continue
    
    if failed_count > 0:
        logger.info(f"总计{failed_count}行解析失败，已跳过")


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
    
    针对金融文档优化：
    - 政府公告：去除分局标题
    - 年报数据：去除公司名称行
    - 新闻文章：去除来源标识
    """
    if not text:
        return ""
    
    # 按换行或中文句号分割，跳过第一段（通常是标题）
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
    """计算包含度相似度：交集占较小集合的比例。"""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return inter / min_size if min_size > 0 else 0.0


def get_key_phrases(text: str, phrase_len: int = 8) -> Set[str]:
    """提取关键短语，针对金融文档优化。
    
    保留：
    - 金融术语和概念
    - 数字和代码（股票代码、基金代码等）
    - 专业表述
    """
    # 保留数字、中文、英文字母，去除其他符号
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', str(text))
    if len(text) < phrase_len:
        return {text} if text else set()
    return {text[i:i+phrase_len] for i in range(len(text) - phrase_len + 1)}


def key_phrase_overlap(phrases_a: Set[str], phrases_b: Set[str], 
                       min_overlap: int = 10) -> bool:
    """检测两个文档是否有足够多的公共关键短语。"""
    if not phrases_a or not phrases_b:
        return False
    common = phrases_a & phrases_b
    return len(common) >= min_overlap


def is_financial_structured_data(content: Union[str, dict]) -> bool:
    """判断是否为金融结构化数据（股票行情、基金数据等）。"""
    if isinstance(content, dict):
        # 检查是否包含金融数据的关键字段
        financial_keys = {
            '股票代码', '基金代码', '交易日期', '交易日', '收盘价', 
            '开盘价', '最高价', '最低价', '成交量', '成交金额',
            '单位净值', '累计单位净值', '资产净值'
        }
        content_keys = set(content.keys())
        return len(financial_keys & content_keys) >= 2
    return False


def is_too_similar_text(a: str, b: str,
                        jaccard_threshold: float = 0.5,
                        containment_threshold: float = 0.8,
                        phrase_overlap_threshold: int = 10,
                        ngram_size: int = 3,
                        phrase_len: int = 8) -> bool:
    """基于字符级 n-gram 和关键短语的中文文本相似度检测。"""
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
    """为近似去重构造一个粗粒度 bucket key（基于高频 n-gram）。"""
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
    """计算内容的哈希指纹，用于快速精确去重。"""
    body = extract_body_for_dedup(text)
    norm = normalize_text(body)
    # 只取正文的核心部分（前500字符）计算哈希，避免尾部差异
    core = norm[:500] if len(norm) > 500 else norm
    return hashlib.md5(core.encode('utf-8')).hexdigest()


def compute_simhash(text: str, ngram_size: int = 3, hash_bits: int = 64) -> int:
    """计算 SimHash 指纹，用于快速近似去重。"""
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


# 针对金融文档优化的噪声过滤模式
FOOTER_PATTERNS = [
    # 政府网站页脚
    r"法律声明\s*\|[\s\S]*?京icp备[0-9a-zA-Z\-]+号[\s\S]*?京公网安备[0-9]+号",
    r"国家外汇管理局[\S ]+分局主办[\s\S]*?网站标识码bm[0-9]+",
    r"网站标识码bm[0-9]+",
    r"主办单位：[\s\S]*?网站标识码：[\s\S]*?",
    # HTML残留
    r"<[^>]*>",
    r"&[a-zA-Z]+;",
    # 表格残留标记
    r'border="[^"]*"',
    r'colspan="[^"]*"',
    r'rowspan="[^"]*"',
    r'</?t[rdh][^>]*>',
]

LINE_DROP_KEYWORDS = [
    "举报本回复", "模拟交易:模拟炒股免费实操交易技能", "微牛证券", "webull",
    "版权所有", "免责声明", "风险提示", "投资有风险", "入市需谨慎",
    "本网站不对所刊载的信息", "仅供参考", "不构成投资建议",
    "设为首页", "加入收藏", "联系我们", "网站地图"
]

# 金融广告和垃圾信息关键词
SPAM_PATTERNS = [
    r"模拟炒股.*?免费.*?实操",
    r"开户.*?佣金.*?万.*?",
    r"股票.*?推荐.*?QQ群",
    r"微信.*?拉群.*?股票",
    r"免费.*?诊股.*?服务",
]


def strip_node_prefix(text: str) -> str:
    """去掉 pkl 抽取时生成的 'node id: ... text:' 前缀。"""
    return re.sub(r"^node id:\s*\S+\s+text:\s*", "", text.strip(), flags=re.IGNORECASE)


def remove_noise(text: str) -> str:
    """移除网页页脚、明显广告/导航等噪声，针对金融文档优化。"""
    # 页脚类整体模式
    for pat in FOOTER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 金融垃圾信息
    for pat in SPAM_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 行级噪声：按换行切分，过滤含关键词的行
    parts = re.split(r"[\n\r]", text)
    cleaned_lines: List[str] = []
    for line in parts:
        line = line.strip()
        if not line:
            continue
        if any(k in line for k in LINE_DROP_KEYWORDS):
            continue
        # 过滤纯符号行
        if re.match(r'^[^\u4e00-\u9fff\w]*$', line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return text.strip()


def chinese_ratio(text: str) -> float:
    """计算中文字符比例。"""
    if not text:
        return 0.0
    total = len(text)
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cn / total if total > 0 else 0.0


def is_financial_content(text: str) -> bool:
    """判断是否为有效的金融内容。"""
    financial_keywords = {
        # 基础金融词汇
        '股票', '基金', '债券', '期货', '期权', '外汇', '保险', '银行',
        '证券', '投资', '理财', '融资', '贷款', '信贷', '资产', '负债',
        # 市场相关
        '交易', '行情', '涨跌', '收盘', '开盘', '成交量', '市值', '估值',
        # 监管机构
        '证监会', '银保监会', '央行', '外汇管理局', '金融委',
        # 财务指标
        '净利润', '营业收入', '资产负债率', '净资产收益率', '市盈率',
        # 政策法规
        '货币政策', '财政政策', '金融监管', '风险管理'
    }
    
    # 检查是否包含金融关键词
    keyword_count = sum(1 for keyword in financial_keywords if keyword in text)
    return keyword_count >= 2 or len(text) > 500  # 包含2个以上金融词汇或长度足够


def clean_record(record: Dict) -> List[Dict]:
    """对单条样本做清洗与拆分，针对金融文档优化。"""
    rid = record.get("id")
    contents = record.get("contents")
    metadata = record.get("metadata", {})

    # 结构化金融数据：保持原样返回
    if is_financial_structured_data(contents):
        return [record]

    # 文本数据：清洗 & 验证
    text = str(contents or "").strip()
    if not text:
        return []

    text = strip_node_prefix(text)
    text = remove_noise(text)

    # 过滤条件优化
    # 1. 极短文本
    if len(text) < 100:
        return []
    
    # 2. 中文比例过低
    if chinese_ratio(text) < 0.3:
        return []
    
    # 3. 非金融内容（可选，根据需要启用）
    # if not is_financial_content(text):
    #     return []

    # 4. 专门过滤 "只有标题+页脚" 的短文本
    if (
        len(text) < 200
        and any(keyword in text for keyword in ["法律声明", "京icp备", "联系我们"])
    ):
        return []

    # 构建清洗后的文档
    new_doc: Dict[str, Union[str, Dict]] = {
        "id": rid,
        "contents": text,
    }
    
    if isinstance(metadata, dict):
        new_doc["metadata"] = dict(metadata)
        # 添加清洗标记
        new_doc["metadata"]["cleaned"] = True
        new_doc["metadata"]["original_length"] = len(str(contents))
        new_doc["metadata"]["cleaned_length"] = len(text)

    return [new_doc]


def process_file(input_path: str, output_path: str, 
                 simhash_threshold: int = 8,
                 jaccard_threshold: float = 0.4,  # 降低阈值，更严格去重
                 containment_threshold: float = 0.8,
                 phrase_overlap_threshold: int = 5,  # 降低阈值，更严格去重
                 phrase_len: int = 8) -> None:
    """处理文件，进行多层去重，针对金融文档优化。"""
    
    logger.info(f"开始处理文件: {input_path}")
    logger.info(f"去重参数: jaccard_threshold={jaccard_threshold}, phrase_overlap={phrase_overlap_threshold}")
    
    cleaned: List[Dict] = []
    seen_texts: Set[str] = set()
    seen_hashes: Set[str] = set()
    simhash_index: Dict[int, List[Tuple[int, str]]] = {}
    buckets: Dict[str, List[Tuple[str, Set[str]]]] = {}
    
    dup_stats = {"hash": 0, "exact": 0, "simhash": 0, "ngram": 0, "phrase": 0}
    process_stats = {"total": 0, "cleaned": 0, "structured": 0, "filtered": 0}

    for rec in tqdm(load_jsonl(input_path), desc="清洗与去重", unit="doc"):
        process_stats["total"] += 1
        docs = clean_record(rec)
        
        if not docs:
            process_stats["filtered"] += 1
            continue
            
        for d in docs:
            contents = d.get("contents")
            
            # 结构化数据直接保留
            if isinstance(contents, dict):
                cleaned.append(d)
                process_stats["structured"] += 1
                continue
            
            # 文本去重处理
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
            
            # 提取正文用于近似去重
            body_norm = normalize_text(extract_body_for_dedup(contents))
            text_for_dedup = body_norm if body_norm else norm
            
            # 第三层：SimHash 近似去重
            simhash = compute_simhash(text_for_dedup)
            simhash_bucket = simhash >> 48
            is_simhash_dup = False
            if simhash_bucket in simhash_index:
                for stored_hash, _ in simhash_index[simhash_bucket]:
                    if is_simhash_similar(simhash, stored_hash, simhash_threshold):
                        is_simhash_dup = True
                        break
            if is_simhash_dup:
                dup_stats["simhash"] += 1
                continue
            
            # 预计算关键短语
            phrases = get_key_phrases(text_for_dedup, phrase_len)
            
            # 第四、五层：N-gram 和短语重叠去重
            bucket_key = make_bucket_key(text_for_dedup)
            is_dup = False
            dup_type = None
            
            if bucket_key:
                bucket_list = buckets.setdefault(bucket_key, [])
                for kept_text, kept_phrases in bucket_list:
                    # N-gram 相似度检查
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
                    
                    # 关键短语重叠检查
                    if key_phrase_overlap(phrases, kept_phrases, phrase_overlap_threshold):
                        is_dup = True
                        dup_type = "phrase"
                        break
            
            if is_dup:
                dup_stats[dup_type] += 1
                continue
            
            # 添加到索引
            if bucket_key:
                buckets[bucket_key].append((text_for_dedup, phrases))
            
            # 通过所有检查，保留文档
            seen_hashes.add(content_hash)
            seen_texts.add(norm)
            simhash_index.setdefault(simhash_bucket, []).append((simhash, text_for_dedup))
            
            cleaned.append(d)
            process_stats["cleaned"] += 1
    
    # 输出统计信息
    total_dup = sum(dup_stats.values())
    logger.info(f"\n处理统计:")
    logger.info(f"  - 总输入文档: {process_stats['total']}")
    logger.info(f"  - 结构化数据: {process_stats['structured']}")
    logger.info(f"  - 清洗过滤: {process_stats['filtered']}")
    logger.info(f"  - 保留文档: {len(cleaned)}")
    
    logger.info(f"\n去重统计:")
    logger.info(f"  - 哈希去重: {dup_stats['hash']}")
    logger.info(f"  - 精确去重: {dup_stats['exact']}")
    logger.info(f"  - SimHash去重: {dup_stats['simhash']}")
    logger.info(f"  - N-gram去重: {dup_stats['ngram']}")
    logger.info(f"  - 短语重叠去重: {dup_stats['phrase']}")
    logger.info(f"  - 总去重数: {total_dup}")
    logger.info(f"  - 去重率: {total_dup/process_stats['total']*100:.2f}%")
    
    save_jsonl(output_path, cleaned)
    logger.info(f"清洗完成，输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="金融文档清洗与去重（五层去重策略）")
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
        default=0.4,
        help="Jaccard 相似度阈值（默认 0.4，更严格）",
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
        default=5,
        help="关键短语重叠数量阈值（默认 5，更严格）",
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


if __name__ == "__main__":
    main()
