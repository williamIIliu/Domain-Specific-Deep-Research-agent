"""对 RAG 数据集中的 relevant_contents 进行去重。

问题：检索到的相关文档中存在高度相似的内容（如不同公司年报的相同模板段落），
这会降低 RAG 的多样性和质量。

解决方案：使用字符级 n-gram 相似度对 relevant_contents 内部进行去重，
保留最相关（score 最高）的文档，去除相似的冗余文档。

使用方法:
    python src/preprocess/dedup_rag_context.py \
        --input datasets/rag_test/context_relevant.jsonl \
        --output datasets/rag_test/context_relevant_dedup.jsonl \
        --jaccard_threshold 0.5 \
        --min_relevant 1
"""

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Set, Any, Optional
from tqdm import tqdm


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """提取字符级 n-gram 集合（适合中文）。"""
    # 去除空白和标点
    text = re.sub(r'[\s\d\.,;:!?，。；：！？、""''（）\[\]【】]', '', str(text))
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


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
    """提取关键短语（连续字符片段），用于检测模板化内容。"""
    # 去除空白和标点，保留核心文字
    text = re.sub(r'[\s\d\.,;:!?，。；：！？、\"\"''（）\[\]【】\n\r]', '', str(text))
    if len(text) < phrase_len:
        return {text} if text else set()
    # 提取所有长度为 phrase_len 的连续片段
    return {text[i:i+phrase_len] for i in range(len(text) - phrase_len + 1)}


def key_phrase_overlap(phrases_a: Set[str], phrases_b: Set[str], 
                       min_overlap: int = 3) -> bool:
    """检测两个文档是否有足够多的公共关键短语。
    
    用于检测模板化内容（如年报中的固定表述）。
    """
    if not phrases_a or not phrases_b:
        return False
    common = phrases_a & phrases_b
    return len(common) >= min_overlap


def is_similar(ngrams_a: Set[str], ngrams_b: Set[str],
               phrases_a: Set[str], phrases_b: Set[str],
               jaccard_threshold: float = 0.5,
               containment_threshold: float = 0.8,
               phrase_overlap_threshold: int = 5) -> bool:
    """判断两个文档是否相似。
    
    使用三种策略：
    1. Jaccard 相似度
    2. 包含度相似度
    3. 关键短语重叠（检测模板化内容）
    """
    # Jaccard 相似度
    jacc = jaccard_similarity(ngrams_a, ngrams_b)
    if jacc >= jaccard_threshold:
        return True
    
    # 包含度相似度
    cont = containment_similarity(ngrams_a, ngrams_b)
    if cont >= containment_threshold:
        return True
    
    # 关键短语重叠（检测模板化内容）
    if key_phrase_overlap(phrases_a, phrases_b, phrase_overlap_threshold):
        return True
    
    return False


def extract_text_content(item: Any) -> str:
    """从 relevant_contents 项中提取文本内容。"""
    if isinstance(item, dict):
        contents = item.get("contents", "")
        if isinstance(contents, dict):
            # 结构化数据，转为字符串
            return json.dumps(contents, ensure_ascii=False)
        return str(contents)
    return str(item)


def dedup_relevant_contents(relevant_contents: List[Dict],
                            jaccard_threshold: float = 0.5,
                            containment_threshold: float = 0.8,
                            ngram_size: int = 3,
                            phrase_len: int = 8,
                            phrase_overlap_threshold: int = 5,
                            min_relevant: int = 1) -> List[Dict]:
    """对 relevant_contents 列表进行去重。
    
    策略：
    1. 按 score 降序排列（保留最相关的）
    2. 依次检查每个文档，如果与已保留的文档相似则跳过
    3. 至少保留 min_relevant 个文档
    
    相似度判断使用三种方法：
    - Jaccard 相似度（基于 n-gram）
    - 包含度相似度（检测子集关系）
    - 关键短语重叠（检测模板化内容）
    
    Args:
        relevant_contents: 相关文档列表
        jaccard_threshold: Jaccard 相似度阈值
        containment_threshold: 包含度相似度阈值
        ngram_size: n-gram 大小
        phrase_len: 关键短语长度
        phrase_overlap_threshold: 关键短语重叠数量阈值
        min_relevant: 最少保留的文档数
    
    Returns:
        去重后的文档列表
    """
    if not relevant_contents:
        return []
    
    # 按 score 降序排列
    sorted_contents = sorted(relevant_contents, 
                             key=lambda x: x.get("score", 0), 
                             reverse=True)
    
    kept: List[Dict] = []
    kept_ngrams: List[Set[str]] = []
    kept_phrases: List[Set[str]] = []
    
    for item in sorted_contents:
        text = extract_text_content(item)
        if not text:
            continue
        
        ngrams = get_char_ngrams(text, ngram_size)
        phrases = get_key_phrases(text, phrase_len)
        if not ngrams:
            continue
        
        # 检查是否与已保留的文档相似
        is_dup = False
        for kept_ng, kept_ph in zip(kept_ngrams, kept_phrases):
            if is_similar(ngrams, kept_ng, phrases, kept_ph,
                          jaccard_threshold, containment_threshold,
                          phrase_overlap_threshold):
                is_dup = True
                break
        
        if not is_dup:
            kept.append(item)
            kept_ngrams.append(ngrams)
            kept_phrases.append(phrases)
    
    # 确保至少保留 min_relevant 个文档
    if len(kept) < min_relevant and len(sorted_contents) >= min_relevant:
        # 补充一些文档（即使相似）
        for item in sorted_contents:
            if item not in kept:
                kept.append(item)
                if len(kept) >= min_relevant:
                    break
    
    # 重新按原始 rank 排序
    kept.sort(key=lambda x: x.get("rank", float('inf')))
    
    return kept


def process_record(record: Dict,
                   jaccard_threshold: float,
                   containment_threshold: float,
                   ngram_size: int,
                   phrase_len: int,
                   phrase_overlap_threshold: int,
                   min_relevant: int) -> Dict:
    """处理单条记录，对 relevant_contents 去重。"""
    result = dict(record)
    
    relevant_contents = record.get("relevant_contents", [])
    if relevant_contents:
        deduped = dedup_relevant_contents(
            relevant_contents,
            jaccard_threshold=jaccard_threshold,
            containment_threshold=containment_threshold,
            ngram_size=ngram_size,
            phrase_len=phrase_len,
            phrase_overlap_threshold=phrase_overlap_threshold,
            min_relevant=min_relevant
        )
        result["relevant_contents"] = deduped
    
    return result


def main():
    parser = argparse.ArgumentParser(description="对 RAG 数据集的 relevant_contents 进行去重")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.5,
        help="Jaccard 相似度阈值，超过此值视为重复（默认 0.5）",
    )
    parser.add_argument(
        "--containment_threshold",
        type=float,
        default=0.8,
        help="包含度相似度阈值，超过此值视为重复（默认 0.8）",
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=3,
        help="n-gram 大小（默认 3）",
    )
    parser.add_argument(
        "--min_relevant",
        type=int,
        default=1,
        help="每条记录最少保留的相关文档数（默认 1）",
    )
    parser.add_argument(
        "--phrase_len",
        type=int,
        default=8,
        help="关键短语长度（默认 8）",
    )
    parser.add_argument(
        "--phrase_overlap",
        type=int,
        default=5,
        help="关键短语重叠数量阈值，超过此值视为模板化重复（默认 5）",
    )
    args = parser.parse_args()
    
    # 统计信息
    total_records = 0
    total_relevant_before = 0
    total_relevant_after = 0
    
    # 处理文件
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="处理中"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            relevant_before = len(record.get("relevant_contents", []))
            
            processed = process_record(
                record,
                jaccard_threshold=args.jaccard_threshold,
                containment_threshold=args.containment_threshold,
                ngram_size=args.ngram_size,
                phrase_len=args.phrase_len,
                phrase_overlap_threshold=args.phrase_overlap,
                min_relevant=args.min_relevant
            )
            
            relevant_after = len(processed.get("relevant_contents", []))
            
            total_records += 1
            total_relevant_before += relevant_before
            total_relevant_after += relevant_after
            
            fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
    
    # 输出统计
    print(f"\n处理完成:")
    print(f"  - 总记录数: {total_records}")
    print(f"  - 去重前 relevant_contents 总数: {total_relevant_before}")
    print(f"  - 去重后 relevant_contents 总数: {total_relevant_after}")
    print(f"  - 去除重复数: {total_relevant_before - total_relevant_after}")
    if total_relevant_before > 0:
        ratio = (total_relevant_before - total_relevant_after) / total_relevant_before * 100
        print(f"  - 去重比例: {ratio:.2f}%")


if __name__ == "__main__":
    main()
