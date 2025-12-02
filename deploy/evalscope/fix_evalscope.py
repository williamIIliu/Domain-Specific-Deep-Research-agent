import os
import sys
import re
import sqlite3

try:
    import evalscope.perf.benchmark
except ImportError:
    print("❌ 错误: 未找到 evalscope 库。请确保你在安装了 evalscope 的环境中运行此脚本 (例如激活 .venv)。")
    sys.exit(1)

# 定位库文件路径
benchmark_file = evalscope.perf.benchmark.__file__
print(f"📍 定位到 benchmark 文件: {benchmark_file}")

with open(benchmark_file, 'r') as f:
    content = f.read()

# 检查是否已经打过补丁
if "EVALSCOPE_PERF_SKIP_DB" in content:
    print("✅ benchmark.py 已经支持 EVALSCOPE_PERF_SKIP_DB，跳过此文件。")
    # sys.exit(0) # 不要退出，继续检查下一个文件
else:
    # 正则匹配数据库连接代码块
    # 目标: with sqlite3.connect(result_db_path) as con:
    pattern = r"(\n\s+)with sqlite3\.connect\(result_db_path\) as con:"
    match = re.search(pattern, content)

    if not match:
        print("❌ 错误: 无法在 benchmark.py 中定位到数据库连接代码，可能是版本不匹配。")
        sys.exit(1)

    full_match = match.group(0)
    indent = match.group(1).replace('\n', '') # 获取缩进

    print(f"🔧 正在应用补丁...")

    # 构造补丁代码：如果环境变量存在，则使用 Mock 对象伪装成数据库连接
    patch_code = f"""
{indent}import os
{indent}# PATCH START
{indent}skip_db = os.environ.get('EVALSCOPE_PERF_SKIP_DB', '0') == '1'
{indent}if skip_db:
{indent}    class MockCon:
{indent}        def cursor(self): return self
{indent}        def execute(self, *args, **kwargs): pass
{indent}        def commit(self): pass
{indent}        def close(self): pass
{indent}        def __enter__(self): return self
{indent}        def __exit__(self, *args): pass
{indent}    cm = MockCon()
{indent}else:
{indent}    cm = sqlite3.connect(result_db_path)
{indent}# PATCH END

{indent}with cm as con:"""

    # 替换原始内容
    new_content = content.replace(full_match, patch_code)

    # 写入文件
    with open(benchmark_file, 'w') as f:
        f.write(new_content)
    print("✅ benchmark.py 已修补。")

# -------------------------------------------------------------------------
# 第二步：修补 db_util.py
# -------------------------------------------------------------------------
try:
    import evalscope.perf.utils.db_util
except ImportError:
    print("⚠️ 警告: 未找到 evalscope.perf.utils.db_util，跳过第二步修补。")
    sys.exit(0)

db_util_file = evalscope.perf.utils.db_util.__file__
print(f"📍 定位到 db_util 文件: {db_util_file}")

with open(db_util_file, 'r') as f:
    db_content = f.read()

if "EVALSCOPE_PERF_SKIP_DB" in db_content:
    print("✅ db_util.py 已经支持 EVALSCOPE_PERF_SKIP_DB，无需重复修补。")
else:
    # 目标: def summary_result(args: Arguments, metrics: BenchmarkMetrics, result_db_path: str):
    # 我们只匹配 def summary_result
    db_pattern = r"def summary_result\(.*?\):"
    db_match = re.search(db_pattern, db_content, re.DOTALL)

    if db_match:
        print("🔧 正在修补 db_util.py ...")
        full_db_match = db_match.group(0)
        
        # 在函数体开头插入检查
        # 我们假设函数定义后面是换行和缩进
        # 为了通用，我们在函数定义后直接插入
        
        # 找到函数定义后的冒号
        end_idx = db_match.end()
        
        # 构造补丁
        # 我们需要引入 os，但 db_util 可能没有 import os。
        # 我们在文件头部检查是否导入 os
        if "import os" not in db_content:
            db_content = "import os\n" + db_content
            # 重新定位因为增加了一行
            db_match = re.search(db_pattern, db_content, re.DOTALL)
            end_idx = db_match.end()

        # 插入逻辑
        # 假设标准缩进是 4 个空格
        patch_logic = "\n    if os.environ.get('EVALSCOPE_PERF_SKIP_DB', '0') == '1':\n        return {}, {}\n"
        
        new_db_content = db_content[:end_idx] + patch_logic + db_content[end_idx:]
        
        with open(db_util_file, 'w') as f:
            f.write(new_db_content)
        print("✅ db_util.py 已修补。")
    else:
        print("❌ 错误: 无法在 db_util.py 中定位到 summary_result 函数。")

print("✅ 成功！所有补丁已应用。")