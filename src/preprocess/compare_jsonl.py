import json
import argparse
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import os

def load_jsonl_to_dict(file_path: str) -> Dict[str, Dict]:
    """加载JSONL文件到字典，以id为键"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_id = record.get('id')
                if record_id:
                    data[record_id] = record
                else:
                    print(f"警告：第{line_num}行缺少id字段")
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
    return data

def save_jsonl(file_path: str, records: List[Dict]):
    """保存记录到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_text_report(file_path: str, content: str):
    """保存文本报告"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def analyze_content_changes(old_record: Dict, new_record: Dict) -> Dict:
    """分析单条记录的内容变化"""
    changes = {}
    
    old_content = old_record.get('contents', '')
    new_content = new_record.get('contents', '')
    
    if old_content != new_content:
        changes['content_changed'] = True
        changes['old_content_type'] = type(old_content).__name__
        changes['new_content_type'] = type(new_content).__name__
        
        if isinstance(old_content, str) and isinstance(new_content, str):
            changes['old_length'] = len(old_content)
            changes['new_length'] = len(new_content)
            changes['length_diff'] = len(new_content) - len(old_content)
        
        # 检查是否是噪声清理
        if isinstance(old_content, str) and isinstance(new_content, str):
            if len(new_content) < len(old_content):
                changes['likely_noise_removal'] = True
    
    # 检查metadata变化
    old_meta = old_record.get('metadata', {})
    new_meta = new_record.get('metadata', {})
    
    if old_meta != new_meta:
        changes['metadata_changed'] = True
        new_keys = set(new_meta.keys()) - set(old_meta.keys())
        if new_keys:
            changes['new_metadata_keys'] = list(new_keys)
    
    return changes

def compare_jsonl_files(old_file: str, new_file: str, output_dir: str):
    """比较两个JSONL文件的差异"""
    
    print(f"加载旧文件: {old_file}")
    old_data = load_jsonl_to_dict(old_file)
    print(f"旧文件记录数: {len(old_data)}")
    
    print(f"加载新文件: {new_file}")
    new_data = load_jsonl_to_dict(new_file)
    print(f"新文件记录数: {len(new_data)}")
    
    # 获取ID集合
    old_ids = set(old_data.keys())
    new_ids = set(new_data.keys())
    
    # 分析差异
    only_in_old = old_ids - new_ids  # 被删除的记录
    only_in_new = new_ids - old_ids  # 新增的记录
    common_ids = old_ids & new_ids   # 共同存在的记录
    
    # 统计信息
    stats = {
        'old_total': len(old_data),
        'new_total': len(new_data),
        'deleted_count': len(only_in_old),
        'added_count': len(only_in_new),
        'common_count': len(common_ids),
        'changed_count': 0,
    }
    
    # 分析内容变化
    changed_records = []
    unchanged_records = []
    
    for record_id in common_ids:
        old_record = old_data[record_id]
        new_record = new_data[record_id]
        
        changes = analyze_content_changes(old_record, new_record)
        
        if changes:
            stats['changed_count'] += 1
            changed_records.append({
                'id': record_id,
                'changes': changes,
                'old_record': old_record,
                'new_record': new_record
            })
        else:
            unchanged_records.append(new_record)
    
    # 保存被删除的记录
    deleted_records = [old_data[rid] for rid in only_in_old]
    if deleted_records:
        save_jsonl(f"{output_dir}/deleted_records.jsonl", deleted_records)
        print(f"保存 {len(deleted_records)} 条被删除的记录到: {output_dir}/deleted_records.jsonl")
    
    # 保存新增的记录
    added_records = [new_data[rid] for rid in only_in_new]
    if added_records:
        save_jsonl(f"{output_dir}/added_records.jsonl", added_records)
        print(f"保存 {len(added_records)} 条新增记录到: {output_dir}/added_records.jsonl")
    
    # 保存变化的记录（分别保存旧版本和新版本）
    if changed_records:
        old_changed = [rec['old_record'] for rec in changed_records]
        new_changed = [rec['new_record'] for rec in changed_records]
        
        save_jsonl(f"{output_dir}/changed_records_old.jsonl", old_changed)
        save_jsonl(f"{output_dir}/changed_records_new.jsonl", new_changed)
        print(f"保存 {len(changed_records)} 条变化记录到: {output_dir}/changed_records_*.jsonl")
    
    # 保存详细变化分析
    if changed_records:
        change_analysis = []
        for rec in changed_records:
            change_analysis.append({
                'id': rec['id'],
                'changes': rec['changes']
            })
        save_jsonl(f"{output_dir}/change_analysis.jsonl", change_analysis)
    
    # 生成统计报告
    report = f"""
数据清洗前后对比报告
==================

基本统计:
- 原始文件记录数: {stats['old_total']:,}
- 清洗后记录数: {stats['new_total']:,}
- 记录数变化: {stats['new_total'] - stats['old_total']:+,}

详细分析:
- 被删除记录数: {stats['deleted_count']:,} ({stats['deleted_count']/stats['old_total']*100:.2f}%)
- 新增记录数: {stats['added_count']:,}
- 共同记录数: {stats['common_count']:,}
- 内容变化记录数: {stats['changed_count']:,} ({stats['changed_count']/stats['common_count']*100:.2f}% of common)
- 完全不变记录数: {len(unchanged_records):,}

去重效果:
- 总去重数: {stats['deleted_count']:,}
- 去重率: {stats['deleted_count']/stats['old_total']*100:.2f}%

文件输出:
- deleted_records.jsonl: 被删除的记录
- added_records.jsonl: 新增的记录  
- changed_records_old.jsonl: 变化记录的原始版本
- changed_records_new.jsonl: 变化记录的清洗后版本
- change_analysis.jsonl: 详细变化分析
"""
    
    # 分析变化类型统计
    if changed_records:
        change_types = defaultdict(int)
        noise_removal_count = 0
        metadata_change_count = 0
        
        for rec in changed_records:
            changes = rec['changes']
            if changes.get('likely_noise_removal'):
                noise_removal_count += 1
            if changes.get('metadata_changed'):
                metadata_change_count += 1
            if changes.get('content_changed'):
                change_types['content_changed'] += 1
        
        report += f"""
变化类型分析:
- 内容变化: {change_types['content_changed']:,}
- 疑似噪声清理: {noise_removal_count:,}
- 元数据变化: {metadata_change_count:,}
"""
    
    save_text_report(f"{output_dir}/comparison_report.txt", report)
    print(f"保存对比报告到: {output_dir}/comparison_report.txt")
    
    print("\n" + "="*50)
    print(report)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="比较两个JSONL文件的差异")
    parser.add_argument("--old_file", type=str, required=True, help="旧版JSONL文件路径")
    parser.add_argument("--new_file", type=str, required=True, help="新版JSONL文件路径") 
    parser.add_argument("--output_dir", type=str, default="./comparison_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行比较
    compare_jsonl_files(args.old_file, args.new_file, args.output_dir)

if __name__ == "__main__":
    main()
