# 检索系统与模块微调

## 1. 环境

### 1.1 UV

通过UV可以管理过个项目环境，仅需在不同项目文件夹中安装.venv文件，在运行代码的过程分别进行activate

```shell
run python ···
```

#### 1.1.1 retrieval项目

为了防止出现显存OOM，这里我安装的是faiss cpu进行检索（查询时数据会移动到内存），在search_engine文件夹已经做了进行相应更改。

```
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements_retrieval.txt

export PYTHONPATH="$PWD:$PYTHONPATH"
```



#### 1.1.2 VeRL项目

verl环境安装中flash-attn很容易出问题，所以建议在`./requirements_sglang.txt`中删除flash-attn这一行后，在本地安装。

```
git clone https://github.com/volcengine/verl.git
cd verl
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install -r ./requirements.txt
uv pip install -r ./requirements_sglang.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

安装完sgl之后你会发现安装的torch版本锁定在了torch==2.8，然而能够支持的flash-attn只有`flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`这一版本，所以只能安装python=3.12

手动安装

```
mkdir -p pkgs && cd pkgs
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```



### 1.2 数据集下载

google drive

### 1.3 预训练权重

```
modelscope download --model Qwen/Qwen3-Embedding-0.6B  --local_dir ./pretrain_models/embedding/Qwen3-Embedding-0.6B
modelscope download --model Qwen/Qwen3-0.6B  --local_dir ./pretrain_models/generator/Qwen3-0.6B #测试
modelscope download --model Qwen/Qwen3-8B  --local_dir ./pretrain_models/generator/Qwen3-8B
```

### 1.4 数据集清洗

#### 1.4.1 清洗策略

项目采用五层去重策略对金融文档进行清洗，针对金融领域的特点进行了专门优化。

**去重示例**

```json
{"id": "ec470c99-b26b-4a51-9872-5501b142cecf", "contents": "node id: 5d3bff95-7ca0-40f8-bdb2-321816370f33 text: (二)公司近三年（含报告期）的普通股股利分配方案或预案、资本公积金转增股本方案或预案单位：元币种：人民币 (三)以现金方式回购股份计入现金分红的情况 □适用√不适用 (四)报告期内盈利且母公司可供普通股股东分配利润为正，但未提出普通股现金利润分配 方案预案的，公司应当详细披露原因以及未分配利润的用途和使用计划 □适用√不适用 二、承诺事项履行情况 (一)公司实际控制人、股东、关联方、收购人以及公司等承诺相关方在报告期内或持续到报告期内的承诺事项 (二)公司资产或项目存在盈利预测，且报告期仍处在盈利预测期间，公司就资产或项目 是否达到原盈利预测及其原因作出说明 □已达到□未达到√不适用 (三)业绩承诺的完成情况及其对商誉减值测试的影响 □适用√不适用 三、...", "metadata": {"source_file": "000221.pkl"}, "relevant_contents": [{"id": "dd06d209-e6a8-4f73-bb02-468baf94e794", "contents": "node id: e335dddf-cbc4-4b7f-b20a-da3e9f5c17d5 text: (二)公司近三年（含报告期）的普通股股利分配方案或预案、资本公积金转增股本方案或预案 单位：元币种：人民币 (三)以现金方式回购股份计入现金分红的情况 □适用√不适用 (四)报告期内盈利且母公司可供普通股股东分配利润为正，但未提出普通股现金利润分配 方案预案的，公司应当详细披露原因以及未分配利润的用途和使用计划□适用√不适用 二、承诺事项履行情况 (一)公司实际控制人、股东、关联方、收购人以及公司等承诺相关方在报告期内或持续到报告期内的承诺事项 (二)公司资产或项目存在盈利预测，且报告期仍处在盈利预测期间，公司就资产或项目 是否达到原盈利预测及其原因作出说明 □已达到□未达到√不适用 (三)业绩承诺的完成情况及其对商誉减值测试的影响 □适用√不适用 三、报...", "metadata": {}, "score": 0.9486966133117676, "rank": 4}]}
```

**执行命令：**

```shell
python src/preprocess/data_clean_new.py \ 
	   --input datasets/OmniEval-Corpus/all_data_raw.jsonl \ 
	   --output datasets/OmniEval-Corpus/all_data_clean_new2.jsonl \ 
	   --jaccard_threshold 0.4 \ 
	   --phrase_overlap 5 \ 
	   --simhash_threshold 8 \ 
	   --containment_threshold 0.8 \ 
	   --phrase_len 8
```



#### 1.4.2 五层去重机制

- **哈希去重**：基于文档正文前500字符的MD5哈希快速去重
- **精确去重**：完全相同的标准化文本去重
- **SimHash去重**：使用局部敏感哈希检测近似重复（汉明距离≤8）
- **N-gram去重**：基于字符级3-gram的Jaccard相似度（≥0.4）和包含度相似度（≥0.8）
- **短语重叠去重**：检测关键短语重叠（≥5个8字符短语），专门处理模板化内容



#### 1.4.3 噪声过滤

**HTML残留清理：**

```json
{
  "原始": "border=\"1\" ><tr> <td colspan=\"1\" rowspan=\"1\">项目</td> <td colspan=\"1\" rowspan=\"1\">2015年度</td>...",
  "清理后": "项目 2015年度 2014年度 2013年度 净利润 20044.10 16004.82..."
}
```

**页脚噪声去除**：

```
原始: "...法律声明 | 联系我们 | 设为首页 | 加入收藏 京ICP备... 京公网安备..."
清理后: 完全移除此类内容
```

**金融垃圾信息过滤**：

- 模拟炒股广告："模拟交易:模拟炒股免费实操交易技能"
- 证券推广："微牛证券"、"开户佣金万X"等
- 投资群组："股票推荐QQ群"、"微信拉群股票"等



#### 1.4.4 质量控制标准

- **最小长度**：100字符（避免标题碎片）
- **中文比例**：≥30%（确保中文金融内容）
- **金融相关性**：包含≥2个金融关键词或长度>500字符
- **结构化数据保护**：自动识别并保留股票行情、基金数据等结构化信息



#### 1.4.5 清洗效果

处理统计：

- 总输入文档：364,816条
- 结构化数据保留：191,018条
- 清洗过滤：10,015条
- 最终保留：308,346条



去重统计：

- 哈希去重：20,449条
- SimHash去重：9,578条
- N-gram去重：12,386条
- 短语重叠去重：4,042条
- **总去重率：12.73%**



#### 1.4.6 对比清洗数据

<u>因为清洗数据的方式有多种，所以这里也写了脚本来对比不同的清洗方法与原repo方法之间的数据差异。</u>

运行指令：

```shell
python src/preprocess/compare_jsonl.py \ 
       --old_file ./datasets/OmniEval-Corpus/all_data_clean.jsonl \ 
       --new_file ./datasets/OmniEval-Corpus/all_data_clean_new.jsonl \ 
       --output_dir ./datasets/OmniEval-Corpus/comparison_results
```

------



## 2. Q&A双阶段蒸馏

### 2.1 Q&A 双阶段蒸馏的背景与目的

#### 2.1.1 为什么需要Q&A蒸馏？

其实传统的 `Embedding model` 在训练时通常使用的都是简单的 **正负样本对** ，但是这种方式存在以下几个问题：

- **缺乏真实查询场景**：训练数据与实际使用场景不匹配；
- **负样本质量低**：随机负样本无法有效提升模型区分能力；
- **缺乏指令理解**：无法处理复杂的检索意图。

如果使用 Q&A 蒸馏，可以通过构造高质量的 **query-document对** 来解决这些问题。

#### 2.1.2 双阶段设计的原因

- 为什么要分为两个阶段？
  - **第一阶段**：确定"谁会问什么样的问题" --> 解决Query的合理性。
  - **第二阶段**：生成具体的Query --> 解决Query的多样性和质量。
- 通过上面的阶段分离，可以：
  - 提高Query的真实性（符合特定角色特征）；
  - 增加Query的多样性（不同角色、问题类型、难度）；
  - 提升训练效果（更贴近实际使用场景）。

### 2.2、Pipeline Design

参考 **Qwen3 Embedding** 模型的数据制作思路，加入instruct进行指令微调，对于一个document（作为正例）构建一个query，然后选择预训练的Embedding模型召回的多个结果作为负样本。

与BERT这种encoder only模型使用[SOS]token不同，Qwen3 使用每一句话的[EOS]token的最后一层的潜变量作为语句表示，因为是Causal模型,而且加入了instruct。

```
{Instruction} {Query}<|endoftext|>
```

### 2.3 一阶段：context 配置

主要是为了配置提问角色，问题类型，问题难度，生成更贴合真实的数据。

#### 2.3.1 提问角色库构建

根据腾讯的 **personal_hub** 提供人物画像，检索含有 **finance** 关键词的画像人物，然后翻译成中文。

- 有不同职业如记者，金融咨询师，数据科学家，也有不同身份：学生，教授，从业者，政客，以及他们及自己的特点：

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

- 使用 Embedding 对每个文档 content 检索 topk候选的人物，在配置过程中会选择一个提问。



**代码实现：**

- 构建特定人群画像json文件：

  如果使用中文语料，还需要将人物画像描述改成中文。

  ```shell
  # 1. 设置参数
      jsonl_path = "./datasets/persona-hub/persona.jsonl"       # 输入文件路径
      search_keyword = "finance"            # 搜索关键词
      output_path = f"./datasets/persona-hub/{search_keyword}_persona.jsonl"  # 输出文件路径
      
  python src/embedding/distill/find_certain_person.py
  ```

- 搭建语料数据库：

  在 `src/embedding/distill/build_persona_db.py` 中 `main()` ，设置文件路径，这里的 `finance_persona.jsonl` 即为原始`persona`。

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



#### 2.3.2 一阶段任务配置

**作用：**

对于每个文档，system 需要决定：

- **角色选择**：谁最可能对这个文档感兴趣？

  ```shell
  文档：关于量化交易策略的技术报告
  	 → 选择：Python开发者（而非普通投资者）
  ```

- **问题类型确定**：这个角色会问什么类型的问题？

  - `keywords`：关键信息查询；
  - `acquire_knowledge`：知识学习；
  - `summary`：内容总结；
  - `yes_or_no`：是非判断；
  - `background`：背景了解。

- **难度评估**：基于角色背景和文档复杂度。

  - `high_school`：表层理解；
  - `university`：需要专业基础；
  - `phd`：需要深度专业知识。



**整体设计如下：**

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

对每一个context首先召回top5可能会关于这个文档的**提问人**，同时配置会问的**问题类型**以及**问题难度**。

### 2.4 二阶段：Question生成

基于一阶段确定的配置，生成具体的查询语句：

```json
给定一个**角色（Character）**、**文档（Passage）** 和**要求（Requirement）**，请从该角色的视角生成一条查询语句：需满足要求中的所有条件，且该查询能用于检索到指定的文档。最终结果仅以 JSON 格式返回，不包含任何额外文本。

## 格式规则
- **文档（Passage）** 语言：中文
- **角色（Character）** 与 **要求（Requirement）** 描述语言：中文
- **输出限制**：仅输出你认为合适的Generated_Query，而非json文件，无多余文本（如解释、说明、标点外的符号）

```

例如：

- 输入：

  - Character：一位Python开发者，专注于AI在金融领域的应用
  - Passage：某个关于机器学习在风险管理中应用的文档
  - Requirement：问题类型=acquire_knowledge，难度=university

- 输出：

  ```shell
  "如何使用机器学习算法来改进传统的信用风险评估模型？"
  ```



然后运行蒸馏代码：

```shell
# 配置参数
    INPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean.jsonl"
    OUTPUT_JSONL_PATH = "./datasets/OmniEval-Corpus/all_data_clean_query.jsonl"
    PERSONA_INDEX_DIR = "./datasets/persona-hub/finance_persona_index"
    LLM_MODEL = "qwen3-30b-a3b-instruct-2507"#"qwen3-30b-a3b"
    SYSTEM_PROMPT = "你是金融领域的专业分析助手"
        
python src/embedding/distill/distill_complete.py
```

------



## 3. 微调数据集

### 3.1 数据集准备

ms-swift对Qwen3 Embedding进行微调，要求的数据格式如下：

```json
{"messages": [{"role": "user", "content": "2022年4月银行结汇和售汇的具体数据是多少？"}], "positive_messages": [[{"role": "user", "content": ""}]],
"negative_messages": [[{"role": "user", "content": ""]]}
```

#### 样本对

**通过构建三元组的数据集，基于距离或者是对比学习的方法来是模型对垂域知识有更好的理解，经过微调之后显著提升模型的细粒度语义辨识能力，同时能够对于场景的认识会更加深刻**。在之前的QA数据蒸馏中，获得了高质量的正样本Q&A，但是现在需要设置负样本以及负难样本，用来Embedding模型的对比学习训练。

负样本对：

从非当前文本id来找到相应数据蒸馏

难负样本：

使用

#### 对比学习



### 3.2 训练

#### 3.2.1 脚本

使用ms-swift对Embedding模型进行训练,

```shell
python src/embedding/train/merge_lora.py
# LoRA合并
python src/embedding/train/merge_lora.py
```

具体参数

```bash
INFONCE_MASK_FAKE_NEGATIVE=true \ # 过滤掉假负样本，也就是负样本的相似度超过正样本的
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model ../../pretrain_weights/embedding/qwen3-0_6b \
    --task_type embedding \
    --model_type qwen3_emb \
    --torch_dtype bfloat16 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj v_proj o_proj \
    --learning_rate 6e-6 \
    --dataset ../../datasets/OmniEval-Corpus/infonce_neg.jsonl \
    --use_hf true \
    --dataloader_num_workers 2 \
    --split_dataset_ratio 0.05 \
    --num_train_epochs 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --output_dir  ../../pretrain_weights/embedding/qwen3-0_6b_finetune \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero2 \
    --report_to swanlab \
    --swanlab_project embedding_finetune
```



### 3.3 训练评估

#### 3.3.1 评估指标

基于retrieval系统进行检测，

支持 dense, bm25, hybrid 三种检索方式

评估指标: Top3@accuracy, Top5@accuracy, MRR



#### 3.3.2 评估代码

主要是基于`src/embedding/eval/retrieval_eval.py`完成

```shell
bash src/embedding/eval/retrieval_eval.sh
```



## 4. 数据库

其实这里使用Milvus或者是Elastic Search能够加速非常多（3-5倍，并且更好的检索效果），但是考虑到大多数用户没有sudo权限，所以这里使用Faiss作为向量数据库。同时Milvus数据库搭建也有相应代码，感兴趣的同学可以尝试。

### 4.1 稠密检索

#### 4.1.1 Embedding生成

通过bf16、torchrun分布式计算等方式进行加速生成，封装shell脚本如下

```
search_engine/faiss/get_embedding.sh
```

其中Qwen3-Embedding模型使用的是last-token的潜在表示d_model作为一个文本的稠密向量，具体代码：

```
# ---------------------- 3. Embedding计算优化 ----------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    device = last_hidden_states.device
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1].to(device)
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=device),
            sequence_lengths
        ].to(device)


def compute_embeddings(model, batch_dict, device, fp16=True):
    with autocast(enabled=fp16):
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    return embeddings
```

#### 4.1.2 索引生成

```
# 读取文本数据
jsonl_path = "./database/data_with_embedding_shards/all_data_clean_embedding.jsonl"
# FAISS index 生成
save_path = "./datasets/database/faiss_qwen"  # Faiss index 
# BM25 index 生成
json_slice_path = "./datasets/database/bm25_tokenize" # 每一个语料进行分词
index_path = "./datasets/database/bm25" #BM25 index

python search_engine/faiss/index_builder.py
```

### 4.2 稀疏检索

这里使用的是BM25算法，分词器使用的是jieba，但是企业级别的BM25算法一般会使用专业垂域语料库进行分词，同时还会在计算BM25 Score的时候对这些专业词汇进行加权，从而能够避免专业词汇在稠密检索过程中会出现被语义忽视。

#### 4.2.1 改进

1. 饱和词频
   对于饱和词频会限制，当某个词在文档中出现次数超过10次之后会（我们当时是改成了5）停止增加该词的词频，避免关键词堆砌导致出错

2. 专有名词加权
   另外一个创新是结合google的报告中，对于专有名词，金融领域的词表进行加权(赋权是1.2)。（计算时间，是否改源代码，**BM25算法也需要调参**）

3. 保留数字
   而且，对于金融当中存在一些股票基金代码，所以这种文档的数字也需要计算词频：

   ```json
   {"股票代码":"002851","交易日期":"20190125","一级行业名称":"电力设备"}
   ```



#### 4.2.2 环境

使用pyserini进行加速搜索，但是需要安装java环境javac

[Archived OpenJDK GA Releases](https://jdk.java.net/archive/)（下载jdk17以上）

[Ubuntu安装Java环境配置 | 命令详解 | 用户不在sudoers文件中。此事将被报告解决方法-CSDN博客](https://blog.csdn.net/2301_80082921/article/details/147552144)

```
cd pkgs/
wget https://download.oracle.com/otn/java/jdk/11.0.9+11/90cf0d8e399443b8860e362981365b51/JDK-11.0.9_linux-x64_bin.tar.gz #如果不行，需要本地下载解压
tar -zxvf JDK-11.0.9_linux-x64_bin.tar.gz
# 验证能否使用
jdk-11.0.9/bin/java -version
# 环境变量
vim ~/.bashrc
export JAVA_HOME=/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/pkgs/jdk-17.0.2
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

# 另外需要设置一下JVM，先在pkg文件夹find . -name "libjvm.so"，然后把带有server的路径填一下
export JVM_PATH=/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/pkgs/jdk-17.0.2/lib/server/libjvm.so
export LD_LIBRARY_PATH=$(dirname "$JVM_PATH"):$LD_LIBRARY_PATH
source ~/.bashrc
```



### 4.3 混合检索

倒排名算法，之后会优化

### 4.4 部署

- 其中$\alpha$是混合检索的权重，这里面可以越大，越偏向于BM25检索
- retrieval_method决定使用哪一种检索方式

```
 python retrieval_server.py \
   --port 8080 \
   --alpha 0.6 \
   --top_k 5 \
   --index_bm25_path "./datasets/database/bm25" \
   --index_faiss_path "./datasets/database/faiss_qwen/faiss_index.bin" \
   --task_desc "根据给定的搜索查询，检索最相关的段落来回答问题" \
   --jsonl_path "../datasets/OmniEval-Corpus/all_data_clean.jsonl" \
   --log_path "../logs/retrieval" \
   --embedding_model_path ".pretrain_models/embedding/Qwen3-Embedding-0.6B" \
   --retrieval_method "dense"
```

然后运行search_engine/faiss/retrieval_server_test.py

## 5. RAG评估数据蒸馏

#### 相关文档配置

再启动检索系统之后对每条文档做相关文档的配置，会进行相似度的过滤，以及去除本条数据.

这里使用了多种过滤方法

```shell
python src/generator/distill/relevant_context.py 
```







# 生成系统

## 1. SFT

### 1.1 数据集

#### 1.1.1 开源数据集

这里我们使用蚂蚁7月份开源的金融数据来做RL训练数据集。因为：

- 高质量的cot数据，
- 蚂蚁的数据格式是选择题，这有一个好处就是在非数学题类的场景中是难以建立reward的，所以这种方法可以是通过选择的方法来取构建reward。

```bash
cd datasets
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download antgroup/Agentar-DeepFinance-100K --local-dir ./Agentar-DeepFinance-100K
```

#### 1.1.2 蒸馏数据集

在自己业务场景下蒸馏CoT数据集，需要去看不同模型的思维链过程的包装方式。



#### 1.1.3 数据集统计

作为算法工程师首先需要清楚自己的数据特点，这决定了后续模型、训练参数、部署方案的选择。

| 指标           | Question                                       | Answer                                                       |
| :------------- | :--------------------------------------------- | ------------------------------------------------------------ |
| 平均长度       | 394.14                                         | 3,651.71                                                     |
| 最小长度       | 10                                             | 364                                                          |
| 最大长度       | 22,845                                         | 157,661                                                      |
| 中位数         | 120                                            | 1,950                                                        |
| P95            | 2,297                                          | 10,809                                                       |
| 超出阈值样本数 | `> 512`: 9,765 (9.86%) `> 1024`: 6,849 (6.91%) | `> 2048`: 46,891 (47.33%)`> 4096`: 25,245 (25.48%)`> 8192`: 10,288 (10.38%) |

太长的CoT数据并不需要，而且数据集里面有很多是冗余的思考。在选择金融行业模型作为基础模型，同时对本地金融数据做思维链冷启动的训练之后即可，在企业部署则需要精简有效的思考。所以在这里我们使用答案在2048以内的数据集，但是以外的数据，之后仍会拿来做RL让模型进行边界的探索。

代码：

```shell
# SFT格式数据处理
python src/generator/sft/convert2sft.py \
    --input_file datasets/Agentar-DeepFinance-100K/Agentar_DeepFinance_01.jsonl \
    --output_dir datasets/Agentar-DeepFinance-100K \
    --max_length 2048
```



### 1.2 SFT 训练

#### 1.2.1 SFT简介

和pretrain的区别在于训练的概率模型不一样，pretrain的模型是根据上文写下文，而SFT则是根据一段问题思考答案。SFT使用**高质量、有标注**的数据集（通常包含 Prompt 和 Answer 对），对模型进行进一步的训练。其核心目的是让模型学会**指令遵循（Instruction Following）**，即理解用户的意图并给出符合人类预期的回复格式和内容。

#### 1.1.2 训练参数

初始config文件在verl/verl/trainer/config/sft_trainer.yaml

**重点参数**

1. lora rank

   32才会有明显效果

2. max_lenght
   2048，所以这里设置最大长度为2028，同时设置右截断，保证推理初始的一致性

3. use_liger

   **Liger Kernel** 是一个针对 LLM 训练优化的 Triton 内核库，主要优化以下计算：

   | 优化组件               | 作用                                                       |
   | :--------------------- | :--------------------------------------------------------- |
   | **Fused CrossEntropy** | 将 logits 计算和 loss 计算融合，避免存储巨大的 logits 张量 |
   | **Fused RMSNorm**      | 融合 RMSNorm 的多个操作                                    |
   | **Fused RoPE**         | 融合旋转位置编码计算                                       |
   | **Fused SwiGLU**       | 融合 SwiGLU 激活函数                                       |

   通过算子融合（Kernel Fusion），减少中间结果的显存占用和内存带宽消耗。

   - 节省显存

     中等（约 20-30%），cpu offload （可节省 50%+ 参数显存）

   - 速度
     加快计算效果好



然后使用VeRL框架训练，由FSDP加速

```
cd verl
export PYTHONPATH="$PWD:$PYTHONPATH"

sh custom/run_qwen_sft.sh
```

#### 1.1.2 训练参数

### 1.2 SFT效果评估

#### 1.2.1 Perplexity

使用困惑度进行评估

#### 1.2.2 题目转换

选择题变成判断题，或者计算题让模型重新进行回答



## 2. 过程-结果多阶段奖励

### 2.1 数据处理

#### 2.1.1 数据集下载

gsm数据测试：

```shell
export HF_ENDPOINT=https://hf-mirror.com
python ./verl/custom/reward_model/data_process-prm-reward.py --local_save_dir ../datasets/gsm_prm_reward_test/
```

依旧使用蚂蚁数据集，挑选出里面的选择题、数据计算以及明确答案的文本作为数据集

```shell
python -m src.generator.rl.data_process_deepfinance_rl \
    --local_dataset_path datasets/Agentar-DeepFinance-100K/Agentar_DeepFinance_sft.jsonl \
    --local_save_dir datasets/Agentar-DeepFinance-100K/rl/ \
    --max_question_length 512
```

数据统计

| 指标           | Question                                        |
| :------------- | :---------------------------------------------- |
| 平均长度       | 655.87                                          |
| 最小长度       | 269                                             |
| 最大长度       | 23104                                           |
| 中位数         | 375                                             |
| P95            | 2900                                            |
| 超出阈值样本数 | `> 512`: 13,900 (15.11%)`> 1024`: 6,990 (7.60%) |

这些超长的数据来源于

1. **法律合同分析任务** (23K 字符) - 包含完整的法律合同文本
2. **新闻文章摘要任务** (22K, 20K 字符) - 包含多篇新闻文章需要总结
3. **财务报表分析任务**

去掉，我们需要的是RL带来的推理能力



#### 2.1.2 数据集处理

verl自带的数据集格式：

```python
instruction_following = """Solve the following question step by step (no more than 5 steps).  You must wrap your thinking with <think>Step1: ...\nStep2: ...\n</think>,  write the final answer between <answer> and </answer>,  and put the final result inside <|box_start|>result<|box_end|>."""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": "prm_reward",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn
```

#### 2.1.3 必要修改

- instruction
  在这里我们需要进行指令，需要让LLM逐步思考（有明确的关键字眼），这样在RM对于中间过程容易判断思维是否一致，**这里具体的特殊字符的格式需要去看你下载模型的tokenizer_config.json文件**：

  以Qwen3为例，使用的是`<think></think>`以及`<|box_start|><|box_end|>`

```tex
instruction_following="""Solve the following question step by step (no more than 5 steps).  You must wrap your thinking with <think>Step1: ...\nStep2: ...\n</think>,  write the final answer between <answer> and </answer>,  and put the final result inside <|box_start|>result<|box_end|>."""
```

- data source
  这里处理之后的数据格式中"data_source": data_source会决定了reward使用什么奖励函数的脚本，所以需要定义我们自己的data_source，并且在进行utils/reward中进行单独的修改。

  

### 2.2 Reward Model

#### 2.2.1 模型选择

这里使用**FinR1**作为中间过程奖励模型，由上海财经大学训练的金融专业领域模型，具体指标如下：

| Model                         | Parameters | FinQA    | ConvFinQA | Ant_Finance | TFNS     | Finance-Instruct-500k | Average  |
| ----------------------------- | ---------- | -------- | --------- | ----------- | -------- | --------------------- | -------- |
| DeepSeek-R1                   | 671B       | 71.0     | 82.0      | **90.0**    | 78.0     | **70.0**              | **78.2** |
| **Fin-R1**                    | 7B         | **76.0** | **85.0**  | 81.0        | 71.0     | 62.9                  | 75.2     |
| Qwen-2.5-32B-Instruct         | 32B        | 72.0     | 78.0      | 84.0        | 77.0     | 58.0                  | 73.8     |
| DeepSeek-R1-Distill-Qwen-32B  | 32B        | 70.0     | 72.0      | 87.0        | **79.0** | 54.0                  | 72.4     |
| **Fin-R1-SFT**                | 7B         | 73.0     | 81.0      | 76.0        | 68.0     | 61.0                  | 71.9     |
| Qwen-2.5-14B-Instruct         | 14B        | 68.0     | 77.0      | 84.0        | 72.0     | 56.0                  | 71.4     |
| DeepSeek-R1-Distill-Llama-70B | 70B        | 68.0     | 74.0      | 84.0        | 62.0     | 56.0                  | 69.2     |
| DeepSeek-R1-Distill-Qwen-14B  | 14B        | 62.0     | 73.0      | 82.0        | 65.0     | 49.0                  | 66.2     |
| Qwen-2.5-7B-Instruct          | 7B         | 60.0     | 66.0      | 85.0        | 68.0     | 49.0                  | 65.6     |
| DeepSeek-R1-Distill-Qwen-7B   | 7B         | 55.0     | 62.0      | 71.0        | 60.0     | 42.0                  | 58.0     |

```shell
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download SUFE-AIFLM-Lab/Fin-R1 --local-dir ./pretrain_models/reward/Fin-R1 
```

或者使用行业大模型轩辕进行打分。

```
modelscope download --model Duxiaoman-DI/XuanYuan-6B-Chat  --local_dir ./pretrain_models/reward/XuanYuan-6B

转换为GGUF格式
python convert_hf_to_gguf.py ../Domain-Specific-Deep-Research-agent/pretrain_models/reward/XuanYuan-6B/ --outfile ../Domain-Specific-Deep-Research-agent/pretrain_models/reward/XuanYuan-6B_f16.gguf --outtype f16

量化
./build/bin/llama-quantize ../Domain-Specific-Deep-Research-agent/pretrain_models/reward/XuanYuan-6B_f16.gguf ../Domain-Specific-Deep-Research-agent/pretrain_models/reward/XuanYuan-6B_Q4_K_M.gguf Q4_K_M

部署
```



#### 2.2.2 SGLang部署

部署参数参考：[服务器参数 — SGLang 框架](https://docs.sglang.com.cn/backend/server_arguments.html)

运行代码：

```shell
bash verl/custom/reward_model/sglang_client.sh 
```

详细内容

```
python -m sglang.launch_server \
--model-path ./pretrain_models/generator/Qwen3-0.6B \
--trust-remote-code \
--dtype bfloat16 \
--served-model-name reward_model \
--max-total-tokens 1024 \
--tensor-parallel-size 1 \
--mem-fraction-static 0.20 \
--api-key sk-123456 \
--host 0.0.0.0 --port 8060 \
--max-running-requests 4 \
--context-length 1024 
```

主要参数这里可以选择

- 并发数目可以设置gpu_num\*gpu_batch_size\*roolout_num，
- 单次请求最大文本长度为max-total-tokens设置为1024+10，使用RL训练过程中使用的max_response_length，因为输出是得分数字,正常情况仅仅占据1个token。一定要设置这个参数，否则默认使用最大文本长度，KV cache直接拉满。
- 最好单独部署在一张卡上面



#### 2.2.3 评分提示词

需要让模型执行中间过程打分任务，从多个维度进行0-10分打分。

```python
f"""请作为金融领域的专家，评估以下推理过程的质量，给出 0-10 的分数。

问题：{query}

标准答案：
{ground_truth}

模型生成的推理过程：
{think_content}

模型生成的最终答案：
{model_answer}

评分标准（用于评估思维/推理过程的质量，而不是只看最终答案）：
1. 推理过程的一致性：
   - 各步骤之间是否逻辑连贯，上下文是否前后一致，没有自相矛盾。
2. 逐步正确性：
   - 使用的公式是否正确，数据代入是否正确，每一步计算是否存在明显算术错误。
3. 关键要素覆盖度：
   - 是否完整覆盖了解决该金融问题所必须的关键步骤（读取题干数据、选取合适金融公式/方法、代入计算、检查结果合理性等）。
4. 金融业务合理性：
   - 推理过程是否符合基本金融常识和约束（如金额符号、比例范围、时间维度、利率含义等），没有明显违背业务常识的推理。
5. 与标准答案的一致性：
   - 在不直接抄袭标准答案的前提下，思维过程是否能够合理推导出标准答案 {ground_truth}，或者至少朝着正确方向逐步逼近。

请综合以上维度给出一个 0-10 的总评分（0 表示推理过程几乎完全错误或无关，10 表示推理过程非常清晰、严谨且能够正确推导出标准答案）。
只输出一个数字分数（0-10），不要输出任何其他文字。"""
```





### 2.3 源代码

参数可以参考[配置说明 — verl documentation](https://woniu9524.github.io/verl-doc/examples/config.html)

#### 2.3.1 verl源码修改

1. reward计算脚本
   VeRL中的奖励函数代码统一在`verl/verl/utils/reward_score`放置，并且在`verl/verl/utils/reward_score/__init__.py`代码中通过data_source来进行决定使用哪个reward脚本
   所以需要将刚才的data_source指定一个reward代码：

   ```python
    elif data_source in ["prm_reward"]:
           from . import prm_reward
           res = prm_reward.compute_score(solution_str, ground_truth, extra_info,use_process_reward=True)
   ```

2. 获得更加细节的日志信息

   在`verl/verl/utils/reward_score/prm_reward.py`脚本中，我们设置了compute_score函数，然后`verl/verl/utils/reward_score/__init__.py`会调用该函数对数据进行处理，计算reward，然后再返回的时候有这样的设置：

   ```
       if isinstance(res, dict):
           return res
       elif isinstance(res, int | float | bool):
           return float(res)
       else:
           return float(res[0])
   ```

   而VeRL是采用的多成员workers的方法进行管理，reward有单独的manager，我们在`verl/verl/workers/reward_manager/naive.py`可以看到
   ```
         from verl.utils.reward_score import default_compute_score
         ···
          score = self.compute_score(
                   data_source=data_source,
                   solution_str=response_str,
                   ground_truth=ground_truth,
                   extra_info=extra_info,
               )
   
               if isinstance(score, dict):
                   reward = score["score"]
                   # Store the information including original reward
                   for key, value in score.items():
                       reward_extra_info[key].append(value)
               else:
                   reward = score
   ```

   所以在`verl/verl/utils/reward_score/prm_reward.py`中最后返回的结果，就可以按照这种格式进行添加：

   ```
   return {
           "score": final_score,
           "format": weights["format"] * format_reward,
           "answer":  weights["answer"] * answer_reward,
           "process": weights["process"] * process_reward,
           "weights": weights
       }
   ```

   同时`verl/verl/workers/reward_manager/naive.py`需要修改为return_dict: bool = True：

   ```python
   def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
   ```

   然后需要修改运行主函数`verl/verl/trainer/ppo/ray_trainer.py`，在val以及训练阶段中的return_matrix之前。具体看repo。
   ```python
   # 在 return metric_dict 之前加：
   if "format" in reward_extra_infos_dict and len(reward_extra_infos_dict["format"]) > 0:
       metric_dict["val-aux/reward_format/mean"] = float(np.mean(reward_extra_infos_dict["format"]))
   if "answer" in reward_extra_infos_dict and len(reward_extra_infos_dict["answer"]) > 0:
       metric_dict["val-aux/reward_answer/mean"] = float(np.mean(reward_extra_infos_dict["answer"]))
   if "progress" in reward_extra_infos_dict and len(reward_extra_infos_dict["progress"]) > 0:
       metric_dict["val-aux/reward_progress/mean"] = float(np.mean(reward_extra_infos_dict["progress"]))
       
       # 训练阶段：把 PRM 子奖励的 batch mean 记到 metrics 里
       if "format" in reward_extra_infos_dict and len(reward_extra_infos_dict["format"]) > 0:
           metrics["critic/reward_format"] = float(np.mean(reward_extra_infos_dict["format"]))
       if "answer" in reward_extra_infos_dict and len(reward_extra_infos_dict["answer"]) > 0:
           metrics["critic/reward_answer"] = float(np.mean(reward_extra_infos_dict["answer"]))
       if "progress" in reward_extra_infos_dict and len(reward_extra_infos_dict["progress"]) > 0:
           metrics["critic/reward_progress"] = float(np.mean(reward_extra_infos_dict["progress"]))
   ```

   3. 提升reward模型的吞吐量

      参考[deepseek-r1复现踩坑系列2: verl的二次开发-reward模块_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TwV1zdEHw/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=38c12c654dac59d97334554c2da5c1e4)

   







#### 2.3.2 reward 函数

同时prm_reward.py的具体内容也将决定着训练结果的好坏。

1. 格式奖励

   对于刚才的`instruction_following`设计，需要有`<think></think>`以及`<|box_start|><|box_end|>`，同时在think中需要将思维链分成一步一步，便于Progress Reward Model判断。

2. 结果奖励
   由于金融数据集很多采用选择题作为可验证奖励，所以这里看答案是否匹配

3. 过程奖励
   从0-10，然后归一化给出得分

4. 奖励分配

   采用0.5+1.0+1.0总分共计2.5分进行训练，便于建立组内优势。



### 2.4 训练

想要在VeRL使用LoRA做RL训练的同学，需要用vllm（sglang会出问题）。

这里要安装vllm0.11.0，不能安装默认的0.8.4否则会用带`cannot import name 'process_weights_after_loading' from 'vllm.model_executor.model_loader.utils`报错。

同样的，版本会锁在torch2.8，所以环境同上。

```
uv venv .venv --python 3.12
source activate .venv/bin/activate
uv pip install -e .
uv pip install -r ./requirements.txt
uv pip install vllm==0.11.0
export PYTHONPATH="$PWD:$PYTHONPATH"

cd pkgs
uv pip install 
```

而且可能还会遇到ray的问题（如果遇到）：
```
bug: Have you run ray on this node?
pip install --upgrade opentelemetry-api opentelemetry-sdk
```



训练脚本

```shell
cd verl
sh ./custom/run_gsm8k_prm.sh
```

#### 2.3.4 结果分析



## 3. RAG工具调用

原来SearchR1使用的reward计算方式使用过exact_match计算模型输出的答案是否是对的，如下：

```python
def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score
```

但是显然这是具有局限性的，对于wiki这种rag数据集虽然有固定的答案，例如人名、地理、金额等等可以进行em的答案，不过对于金融法律等等其他领域这种计算reward方法并不适合，所以我们使用基于语义理解的判定方法进行修正。在`verl/verl/utils/reward_score/search_r1_semantic_match.py`可以看到。

后续的处理主要是加入retrieval系统进行RAG测试集测试。

```
```



# 部署与压测

## 1. llama.cpp

### 1.1 环境

#### 1.1.1 编译

```shell
mkdir build && cd build

cmake .. \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CUBLAS=OFF \
  -DLLAMA_CURL=OFF
  
  # 添加到系统环境
  export PATH="/mypool/lzq/LLM/llama.cpp-master/build/bin:$PATH"
  source ~/.bashrc
```

#### 1.1.2 pyhton环境

```shell
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-first-match
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 1.2 部署

#### 1.2.1 转换为GGUF格式

```
# 是否需要量化：修改--outtype参数
python convert_hf_to_gguf.py ../Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3-8B/ --outfile ../Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3_8B_f16.gguf --outtype f16
```

#### 1.2.2 量化方案选择

```
./build/bin/llama-quantize ../Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3_8B_f16.gguf ../Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3_8B_Q4_K_M.gguf Q4_K_M
```

中间转换过程中的关键部分attn, ffn_down会使用Q6KM，

outtype是输出类型，代表含义：

- q2_k：特定张量（Tensor）采用较高的精度设置，而其他的则保持基础级别。
- q3_k_l、q3_k_m、q3_k_s：这些变体在不同张量上使用不同级别的精度，从而达到性能和效率的平衡。
- q4_0：这是最初的量化方案，使用 4 位精度。
- Q4_1 和 Q4_K_M、Q4_K_S：这些提供了不同程度的准确性和推理速度，适合需要平衡资源使用的场景。
- q5_0、q5_1、q5_k_m、q5_k_s：这些版本在保证更高准确度的同时，会使用更多的资源并且推理速度较慢。
- q6_k 和 q8_0：这些提供了最高的精度，但是因为高资源消耗和慢速度，可能不适合所有用户。
  fp16 和 f32: 不量化，保留原始精度。

其中量化的具体方案不同，导致量化模型的性能也会带来巨大差异：

| 代号         | 每权重位数 | 是否存最小/最大值(scale) | 是否存零点(zero) | 共享量化参数块大小            | 显存相对大小 | 质量(1→5) | 速度(1→5) | 一句话总结                                                |
| ------------ | ---------- | ------------------------ | ---------------- | ----------------------------- | ------------ | --------- | --------- | --------------------------------------------------------- |
| **q4\_0**    | 4 bit      | ✅                        | ❌                | 32 权重一组                   | 100 %        | ⭐⭐        | ⭐⭐⭐⭐⭐     | 最老、最小、最快，也最糙；玩具/CPU 场景够用。             |
| **q4\_1**    | 4 bit      | ✅                        | ✅                | 32 权重一组                   | +3 %         | ⭐⭐⭐       | ⭐⭐⭐⭐      | 比 q4\_0 多存个“零点”，精度↑，体积稍大，速度几乎不变。    |
| **q4\_k\_m** | 4 bit      | ✅                        | ✅                | 256 权重一组 + K-means 优化   | +6 %         | ⭐⭐⭐⭐      | ⭐⭐⭐       | “中庸”首选，体积/质量折中；GPU/CPU 都推荐。               |
| **q4\_k\_s** | 4 bit      | ✅                        | ✅                | 256 权重一组 + K-means 更激进 | +2 %         | ⭐⭐        | ⭐⭐⭐⭐      | 比 q4\_k\_m 再压狠一点，体积≈ q4\_1，质量略降，速度略快。 |

8B的模型转化为GGUF大小15.6G，Q4_K_M为4.78G，而Q4_0为4.54G。Q4_K_M模型量化过程中attn_v.weight, ffn_down.weight会使用Q6_K保证模型效果，而Q4_0则没有。

#### 1.2.3 推理

```
llama-cli -m ../Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3_8B_Q4_K_M.gguf -p "Hello, what is the meaning of life?"

```

## 2. Ollama

### 2.1 环境

运行GGUF格式文件，GUUF文件推理时消耗更少的资源

```
#下载安装包
wget -O ollama-linux-amd64.tgz https://ollama.com/download/ollama-linux-amd64.tgz
tar -zxvf ollama-linux-amd64.tgz

# 查看版本
./bin/ollama -v

# 环境变量
export PATH="/mypool/lzq/LLM/ollama/bin:$PATH"
```

### 2.2 运行模型

常见命令

```shell
ollama list              # 列出所有模型
ollama pull <model-name> # 从模型仓库拉取一个模型
ollama run <model-name>  # 运行一个模型
ollama ps                # 列出所有正在运行的模型
ollama rm <model-name>   # 删除一个模型
```

#### 2.2.1 运行模型

1. 创建Modelfile文件
   无拓展名，用于设置Ollama部署之后的模型名称，运行参数。主要是用于指定模型路径以及封装

2. 运行模型

   先写文件参数，注意这里的**FROM**指定的路径是从**Modelfile文件**所在位置进行计算的，所以最好填相对路径。
   ```
   FROM /mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/pretrain_models/generator/Qwen3_8B_Q4_K_M.gguf
   
   # set the temperature to 0.7
   PARAMETER temperature 0.7
   PARAMETER top_p 0.8
   PARAMETER repeat_penalty 1.05
   TEMPLATE """{{ if .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}{{ if .Prompt }}<|im_start|>user
   {{ .Prompt }}<|im_end|>
   {{ end }}<|im_start|>assistant
   {{ .Response }}<|im_end|>"""
   # set the system message
   SYSTEM """
   You are a helpful assistant.
   """
   ```

   nohup无sudo安装编译实例

   ```shell
   wget https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz
   tar -xf coreutils-9.4.tar.xz && cd coreutils-9.4
   
   ./configure --prefix=$HOME/local --disable-dependency-tracking
   make -j$(nproc)
   make install
   
   echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

   然后运行模型

   ```shell
   # 在运行之前首先要运行server
   nohup ollama serve &
   # 验证是否运行
   curl http://127.0.0.1:11434
   # 进行部署
   ollama create Fin-Search -f ./deploy/ollama/Qwen3-8B_Q4_K_M/Modelfile
   ```

   如果想要修改运行时的端口与命令，需要在serve之前先export

   ```shell
   export OLLAMA_HOST=0.0.0.0:11434
   export CUDA_VISIBLE_DEVICES=0,1 
   ollama serve
   ```

3. 调用

   运行

   ```shell
   curl http://127.0.0.1:11434/api/chat -d '{"model": "Fin-Search", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

   停止运行

   ```shell
   pkill -f "ollama serve"
   # 如果nohup启动，还需要删除
   rm -f ~/.ollama/ollama.pid
   ```

   

## 3. EvalScope

文档资料：[快速上手 | EvalScope](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)

### 2.1 环境

```
uv pip install 'evalscope[all]'
uv pip install 'evalscope[perf]' 压测
```

### 2.2 压力测试

完整的参数配置信息：[参数说明 | EvalScope](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html)

```bash
evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model Fin-Search \
  --url http://127.0.0.1:11434/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --max-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --tokenizer-path ./pretrain_models/generator/Qwen3-8B/ \
  --extra-args '{"ignore_eos": true}' 
```

测试结果：
```txt
Benchmarking summary:
+-----------------------------------+----------+
| Key                               |    Value |
+===================================+==========+
| Time taken for tests (s)          |  55.3146 |
+-----------------------------------+----------+
| Number of concurrency             |   1      |
+-----------------------------------+----------+
| Total requests                    |  10      |
+-----------------------------------+----------+
| Succeed requests                  |  10      |
+-----------------------------------+----------+
| Failed requests                   |   0      |
+-----------------------------------+----------+
| Output token throughput (tok/s)   | 116.697  |
+-----------------------------------+----------+
| Total token throughput (tok/s)    | 219.963  |
+-----------------------------------+----------+
| Request throughput (req/s)        |   0.195  |
+-----------------------------------+----------+
| Average latency (s)               |   5.5304 |
+-----------------------------------+----------+
| Average time to first token (s)   |   0.3121 |
+-----------------------------------+----------+
| Average time per output token (s) |   0.0087 |
+-----------------------------------+----------+
| Average inter-token latency (s)   |   0.0087 |
+-----------------------------------+----------+
| Average input tokens per request  | 529.7    |
+-----------------------------------+----------+
| Average output tokens per request | 598.6    |
+-----------------------------------+----------+
```

### 2.3 能力测试

主要是检测量化或者微调之后模型的通用能力是否下降

```
uv pip install evalscope[opencompass]
```

所需文件，在yaml文件中指定测试数据集的选择

```
eval_backend: OpenCompass
eval_config:
  datasets:
    - mmlu
    - ceval
    - ARC_c
    - gsm8k
  models:
    - openai_api_base: http://127.0.0.1:11434/v1/chat/completions
      path: Fin-Research                                
      temperature: 0.0
```

