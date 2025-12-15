import json


def find_persona_with_keyword(jsonl_file_path, keyword, output_path):
    """
    检索JSONL文件中persona字段包含指定关键词的人物，并保存结果到指定路径
    
    Args:
        jsonl_file_path: JSONL文件路径（如"./persona.jsonl"）
        keyword: 目标关键词
        output_path: 输出文件路径（结果将保存为JSONL格式）
    
    Returns:
        dict: 包含检索统计信息的字典
            {
                'total_count': 搜索到的条目总数,
                'keyword': 搜索关键词,
                'output_path': 结果保存路径
            }
    """
    matched_personas = []
    line_numbers = []  # 记录匹配的行号
    
    # 逐行读取JSONL文件
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):  # 记录行号，方便定位异常
            try:
                # 解析当前行的JSON对象
                persona_dict = json.loads(line.strip())
                # 检查是否存在"persona"字段，且包含目标关键词
                if "persona" in persona_dict:
                    persona_content = persona_dict["persona"].lower()  # 转为小写，避免大小写敏感
                    if keyword.lower() in persona_content:
                        matched_personas.append(persona_dict)
                        line_numbers.append(line_num)
            except json.JSONDecodeError:
                # 处理无效JSON格式的行
                print(f"警告：第{line_num}行不是有效的JSON格式，已跳过")
            except Exception as e:
                # 处理其他未知错误
                print(f"警告：第{line_num}行处理出错，错误信息：{str(e)}，已跳过")
    
    # 保存结果到输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for persona in matched_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + "\n")
    
    # 生成报告
    print("\n" + "="*60)
    print(f"检索关键词：'{keyword}'")
    print(f"输入文件：{jsonl_file_path}")
    print("="*60)
    print(f"✓ 共检索到 {len(matched_personas)} 条匹配记录")
    print(f"✓ 结果已保存到：{output_path}")
    print("="*60)
    
    return {
        'total_count': len(matched_personas),
        'keyword': keyword,
        'output_path': output_path
    }


if __name__ == "__main__":
    # 1. 设置参数
    jsonl_path = "./datasets/persona-hub/persona.jsonl"       # 输入文件路径
    search_keyword = "finance"            # 搜索关键词
    output_path = f"./datasets/persona-hub/{search_keyword}_persona.jsonl"  # 输出文件路径
    
    # 2. 执行检索并保存结果
    result = find_persona_with_keyword(jsonl_path, search_keyword, output_path)
    