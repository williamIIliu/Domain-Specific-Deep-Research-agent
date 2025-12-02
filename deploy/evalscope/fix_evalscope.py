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
    print("✅ evalscope 已经支持 EVALSCOPE_PERF_SKIP_DB，无需重复修补。")
    sys.exit(0)

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

print("✅ 成功！evalscope 已修补。现在你可以使用 EVALSCOPE_PERF_SKIP_DB=1 来跳过数据库写入了。")