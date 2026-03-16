"""
从FiNER数据集中提取命名实体，构建金融领域自定义词典
用于优化jieba分词，提升BM25检索效果
"""
import json
from collections import Counter
from tqdm import tqdm


def extract_entities_from_finer(jsonl_path: str, output_dict_path: str, min_freq: int = 2):
    """
    从FiNER.jsonl中提取所有命名实体，构建自定义词典
    
    Args:
        jsonl_path: FiNER数据集路径
        output_dict_path: 输出词典路径
        min_freq: 最小词频阈值，低于此频率的实体不加入词典
    """
    entity_counter = Counter()
    entity_types = {}  # 记录实体类型
    
    print("正在从FiNER数据集提取实体...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="提取实体"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                # 找到assistant的回复（包含NER结果）
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        try:
                            # 解析NER结果JSON
                            ner_result = json.loads(content)
                            
                            # 遍历所有实体类型
                            for entity_type, entities in ner_result.items():
                                if isinstance(entities, list):
                                    for entity in entities:
                                        if entity and len(entity.strip()) > 1:  # 过滤单字和空字符
                                            entity = entity.strip()
                                            entity_counter[entity] += 1
                                            # 记录实体类型（如果同一实体有多个类型，保留第一个）
                                            if entity not in entity_types:
                                                entity_types[entity] = entity_type
                        except json.JSONDecodeError:
                            continue
            except json.JSONDecodeError:
                continue
    
    print(f"共提取到 {len(entity_counter)} 个不同实体")
    
    # 按频率过滤并排序
    filtered_entities = {
        entity: (count, entity_types.get(entity, "UNKNOWN"))
        for entity, count in entity_counter.items()
        if count >= min_freq
    }
    
    print(f"频率>={min_freq}的实体数量: {len(filtered_entities)}")
    
    # 保存为jieba自定义词典格式: 词语 词频 词性
    # 词性映射：ORG->nr, PRODUCT->n, METRIC->n, TERM->n, TIME->t
    pos_mapping = {
        "ORG": "nr",      # 机构名
        "PRODUCT": "n",   # 产品名
        "METRIC": "n",    # 指标名
        "TERM": "n",      # 术语
        "TIME": "t"       # 时间
    }
    
    with open(output_dict_path, 'w', encoding='utf-8') as f:
        # 按频率降序排序
        sorted_entities = sorted(filtered_entities.items(), key=lambda x: x[1][0], reverse=True)
        for entity, (freq, entity_type) in sorted_entities:
            pos = pos_mapping.get(entity_type, "n")
            # jieba词典格式：词语 词频 词性
            f.write(f"{entity} {freq} {pos}\n")
    
    print(f"✅ 自定义词典已保存至: {output_dict_path}")
    
    # 输出统计信息
    print("\n实体类型分布:")
    type_counter = Counter([t for _, (_, t) in filtered_entities.items()])
    for entity_type, count in type_counter.most_common():
        print(f"  {entity_type}: {count}")
    
    print("\n高频实体示例 (Top 20):")
    for entity, (freq, entity_type) in sorted_entities[:20]:
        print(f"  {entity} ({entity_type}): {freq}次")
    
    return output_dict_path


if __name__ == "__main__":
    finer_path = "datasets/NER/FiNER.jsonl"
    output_path = "datasets/NER/finance_ner_dict.txt"
    
    extract_entities_from_finer(finer_path, output_path, min_freq=2)
