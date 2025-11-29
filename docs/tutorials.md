# 检索系统与模块微调

## 1. 环境

### 1.1 UV

通过UV可以管理过个项目环境，仅需在不同项目文件夹中安装.venv文件，在运行代码的过程中使用

```shell
uv run python ···
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



### 1.3 预训练权重

```
modelscope download --model Qwen/Qwen3-Embedding-0.6B  --local_dir ./pretrain_models/embedding/Qwen3-Embedding-0.6B

modelscope download --model Qwen/Qwen3-8B  --local_dir ./pretrain_models/generator/Qwen3-8B
```



## 2. Q&A双阶段蒸馏

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



## 3. 微调数据集

### 3.1 数据集准备

ms-swift对Qwen3 Embedding进行微淘

标准格式

#### 样本对

### 3.2 训练代码

使用ms-swift对Embedding模型进行训练

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

#### LoRA合并



### 3.3 训练评估

#### 评估指标

基于retrieval系统进行检测，

支持 dense, bm25, hybrid 三种检索方式

评估指标: Top3@accuracy, Top5@accuracy, MRR



#### 评估代码

主要是基于`src/embedding/eval/retrieval_eval.py`完成

```shell
bash src/embedding/eval/retrieval_eval.sh
```



## 4. 数据库

其实这里使用Milvus或者是Elastic Search能够加速非常多（3-5倍，并且更好的检索效果），但是考虑到大多数用户没有sudo权限，所以这里使用Faiss作为向量数据库。同时Milvus数据库搭建也有相应代码，感兴趣的同学可以尝试。

### 稠密检索

#### Embedding生成

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

#### 索引生成

```
# 读取文本数据
jsonl_path = "../datasets/Fin_Corpus/demo_embedding.jsonl"
# FAISS index 生成
save_path = "./datasets/database/faiss_qwen"  # Faiss index 
# BM25 index 生成
json_slice_path = "./datasets/database/bm25_tokenize" # 每一个语料进行分词
index_path = "./datasets/database/bm25" #BM25 index

python search_engine/faiss/index_builder.py
```

### 稀疏检索

这里使用的是BM25算法，分词器使用的是jieba，但是企业级别的BM25算法一般会使用专业垂域语料库进行分词，同时还会在计算BM25 Score的时候对这些专业词汇进行加权，从而能够避免专业词汇在稠密检索过程中会出现被语义忽视。



可以进行的创新：

主要是对于饱和词频会限制，当某个词在文档中出现次数超过10次之后会（我们当时是改成了5）停止增加该词的词频，避免关键词堆砌导致出错。另外一个创新是结合google的报告中，对于专有名词，金融领域的词表进行加权(赋权是1.2)。（计算时间，是否改源代码，**BM25算法也需要调参**）



而且，对于金融当中存在一些股票基金代码，所以这种文档的数字也需要计算词频：

```json
{"股票代码":"002851","交易日期":"20190125","一级行业名称":"电力设备"}
```

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



### 混合检索



### 部署

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



# 生成系统

## SFT

### 数据集

#### 开源数据集

这里我们使用蚂蚁7月份开源的金融数据来做RL训练数据集。因为：

- 高质量的cot数据，
- 蚂蚁的数据格式是选择题，这有一个好处就是在非数学题类的场景中是难以建立reward的，所以这种方法可以是通过选择的方法来取构建reward。

```bash
cd datasets
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download antgroup/Agentar-DeepFinance-100K --local-dir ./Agentar-DeepFinance-100K

# SFT格式数据处理
python .py \
--input_dir datasets/Agentar-DeepFinance-100K \
--output_dir datasets/Agentar-DeepFinance-100K \
--train_ratio 0.95
```

然后使用VeRL框架训练，由FSDP加速

```
cd verl
export PYTHONPATH="$PWD:$PYTHONPATH"

sh custom/run_qwen_05_sp2_liger.sh
```

#### 训练参数

config文件在verl/verl/trainer/config/sft_trainer.yaml可以详细去看

**重点参数**

1. lora rank

   32才会有明显效果

2. max_lenght
   COT推理链条超过2048的很少，所以这里设置最大长度为2028，同时设置右截断，保证推理初始的一致性

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





#### 蒸馏数据集

同时也需要包含自己业务场景下的常见问题，例如：

- 利率是1.2，接两个月需要多少钱，如果还款迟了一个月，需要多交多少钱



## RL

过程-结果多阶段奖励

工具调用奖励
