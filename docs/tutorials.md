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

google drive

### 1.3 预训练权重

```
modelscope download --model Qwen/Qwen3-Embedding-0.6B  --local_dir ./pretrain_models/embedding/Qwen3-Embedding-0.6B
modelscope download --model Qwen/Qwen3-0.6B  --local_dir ./pretrain_models/generator/Qwen3-0.6B #测试
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

```shell
python src/embedding/train/merge_lora.py
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
jsonl_path = "../datasets/Fin_Corpus/demo_embedding.jsonl"
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

#### 1.1.2 训练参数

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



#### 1.1.3 蒸馏数据集

在自己业务场景下蒸馏CoT数据集，需要去看不同模型的思维链过程的包装方式。



## 2. 过程-结果多阶段奖励

### 2.1 数据处理

运行代码：

```shell
export HF_ENDPOINT=https://hf-mirror.com
python ./verl/custom/reward_model/data_process-prm-reward.py --local_save_dir ../datasets/gsm_prm_reward_test/
```

#### 2.1.1 数据集处理

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

#### 2.1.2 必要修改

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

或者使用Qwen-flash进行大模型打分。



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





### 2.3 RL训练

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



#### 2.3.3 训练

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

