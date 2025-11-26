

## 环境

Embedding

```
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

export PYTHONPATH="$PWD:$PYTHONPATH"
```

微调数据集

## 2 Q&A双阶段蒸馏

参考Qwen3 Embedding 模型的数据制作思路，加入instruct进行指令微调，对于一个doument（作为正例）构建一个query，然后选择预训练的Embedding模型召回的多个结果作为负样本。

与BERT这种encoder only模型使用[SOS]token不同，Qwen3 使用每一句话的[EOS]token的最后一层的潜变量作为语句表示，因为是Causal模型,而且加入了instruct。

```
{Instruction} {Query}<|endoftext|>
```

### 2.1 context 配置

主要是为了配置提问角色，问题类型，问题难度，生成更贴合真实的数据。

#### 2.1.1提问角色库

根据腾讯的personal_hub提供人物画像，检索含有finance关键词的画像人物，然后翻译成中文。

- 有不同职业如记者，金融咨询师，数据科学家，也有不同身份：学生，教授，从业者，政客，以及他们及自己的特点

  ```json
  {'persona': '一位非金融行业的成功创业者，认可高管对“ impostor syndrome（冒名顶替综合征）”的看法，希望获取克服自我怀疑的指导'}
  {'persona': '一位年轻人，在金融行业经历了漫长且充满压力的职业生涯后，重新找回了对文学的热爱'}
  {'persona': '一位女性，借助小额信贷支持，成功经营着一家居家编织作坊'}
  {'persona': '一位才华横溢的数据科学家，牵头运营一家专注于伦理消费金融的金融科技初创公司'}
  {'persona': '一位孟加拉国公民，对金融公司持怀疑态度'}
  {'persona': '一名金融专业大学生，渴望获取理财知识，且将观鸟作为共同爱好，乐在其中'}
  {'persona': '一位同届毕业生，正开启企业金融领域的职业生涯，分享相关技巧与经验'}
  {'persona': '一位金融从业者，负责分析和管理临床试验部门的预算'}
  {'persona': '一位支持型的兄弟姐妹，为这位艺术家的首次个人展览提供了资金支持'}
  {'persona': '一名工商管理专业学生，为熟食店提供市场营销与财务策略方面的见解'}
  {'persona': '一位Python开发者，专注于人工智能与机器学习在金融领域的应用'}
  {'persona': '一名本科低年级学生，有意向从事量化金融领域的工作'}
  {'persona': '一位自由记者兼播客主持人，聚焦个人理财策略相关内容'}
  {'persona': '一位财政保守主义者，对财政部长政策的有效性提出质疑'}
  ```

- 使用Embedding对每个文档content检索topk候选的人物，在配置过程中会选择一个提问。

代码实现：

1. 构建特定人群画像json文件

   如果使用中文语料，还需要将人物画像描述改成中文

   ```shell
   # 1. 设置参数
       jsonl_path = "./datasets/persona-hub/persona.jsonl"       # 输入文件路径
       search_keyword = "sport"            # 搜索关键词
       output_path = f"./datasets/persona-hub/{search_keyword}_persona.jsonl"  # 输出文件路径
       
   python src/embedding/distill/find_certain_person.py
   ```

2. 搭建语料数据库

在src/embedding/distill/build_persona_db.py中main()，设置文件路径，这里的finance_persona.jsonl即为原始persona

```shell
# 配置文件路径
    persona_file = "./datasets/persona-hub/finance_persona.jsonl"
    output_dir = "./datasets/persona-hub/finance_persona_index"
# 构建索引 - 使用 Qwen3-Embedding
    builder = PersonaIndexBuilder(
        model_path="./pretrain_models/embedding/Qwen3-Embedding-0.6B",
        max_length=256,
        device="auto"
    )

python src/embedding/distill/build_persona_db.py
```

#### 2.1.2 一阶段任务配置

```json
emd_stage1 = """
给定一段文档（Passage）和一组候选角色（Characters），请从“角色（Characters）”“问题类型（Question_Type）”“难度（Difficulty）”三个维度选择合适选项，并以 JSON 格式返回输出结果。

操作步骤如下：
1. 从候选角色（Characters）中，筛选出1个最可能对该文档感兴趣的角色；
2. 结合该角色的身份特征，确定其可能针对文档提出的“问题类型（Question_Type）”；
3. 参考文档内容复杂度、角色知识背景及问题类型，确定该问题的“难度（Difficulty）”等级。

各维度可选范围说明：
- 角色（Characters）：由输入的候选角色列表提供，仅选择1个；
- 问题类型（Question_Type）：
  - keywords（关键词型）：围绕文档核心信息的关键词查询，如“文档中提到的XX政策发布时间是什么？”；
  - acquire_knowledge（知识获取型）：获取文档中具体知识点的查询，如“请解释文档中‘XX概念’的含义”；
  - summary（总结型）：对文档核心内容的概括查询，如“总结文档关于XX领域的3个核心观点”；
  - yes_or_no（是非判断型）：对文档内容的是非验证，如“文档是否认为XX措施有效？”；
  - background（背景询问型）：关于文档创作背景或关联信息的查询，如“文档作者撰写本文的行业背景是什么？”；
- 难度（Difficulty）：
  - high_school（高中水平）：无需专业知识，仅需理解文档表层信息即可回答；
  - university（大学水平）：需结合基础专业知识（如金融/经济基础概念）分析文档；
  - phd（博士水平）：需深度专业知识（如学术理论、行业前沿动态）解读文档深层逻辑。
```

对每一个context首先召回top5可能会关于这个文档的**提问人**，同时配置会问的**问题类型**以及**问题难度**

#### 2.1.3 二阶段Question生成

```json
给定一个**角色（Character）**、**文档（Passage）** 和**要求（Requirement）**，请从该角色的视角生成一条查询语句：需满足要求中的所有条件，且该查询能用于检索到指定的文档。最终结果仅以 JSON 格式返回，不包含任何额外文本。

## 格式规则
- **文档（Passage）** 语言：中文
- **角色（Character）** 与 **要求（Requirement）** 描述语言：中文
- **输出限制**：仅输出你认为合适的Generated_Query，而非json文件，无多余文本（如解释、说明、标点外的符号）

```

然后运行蒸馏代码

```shell
# 配置参数
    INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean.jsonl"
    OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_query.jsonl"
    PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
    LLM_MODEL = "qwen3-30b-a3b-instruct-2507"#"qwen3-30b-a3b"
    SYSTEM_PROMPT = "你是金融领域的专业分析助手"
        
python src/embedding/distill/distill_complete.py
```



#### 



## 微调数据集



## 数据库