# Domain-Specific Deep Research Agent

<p align="center">
  <b>面向垂直领域的深度问答系统</b>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#项目结构">项目结构</a> •
  <a href="#详细文档">详细文档</a> •
  <a href="#许可证">许可证</a>
</p>

---

一个专为垂直领域（金融、法律、医疗等）设计的深度问答系统。通过数据蒸馏、模型微调、检索增强生成（RAG）和强化学习，将复杂的专业问题转化为准确、可追溯的回答。

## 特性

- 🔍 **混合检索系统** - 支持稠密检索（Dense）、稀疏检索（BM25）和混合检索，基于 Faiss 构建高效向量索引
- 📚 **Q&A 数据蒸馏** - 基于 Persona 的两阶段蒸馏流程，生成高质量的领域问答对
- 🎯 **Embedding 微调** - 使用 InfoNCE 对比学习，提升领域语义理解能力
- 🧠 **生成模型训练** - 支持 SFT 监督微调和基于过程奖励的强化学习（PRM-RL）
- 🚀 **轻量化部署** - 支持 GGUF 量化、Ollama 部署和压力测试

## 快速开始

### 环境安装

详细的安装步骤请参考 **[安装教程](docs/tutorials.md)**，包括：
- UV 环境管理
- VeRL 框架安装
- 预训练权重下载
- 数据集准备

### 最小示例

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/Domain-Specific-Deep-Research-agent.git
cd Domain-Specific-Deep-Research-agent

# 2. 安装检索系统依赖
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements_retrieval.txt

# 3. 下载预训练模型
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./pretrain_models/embedding/Qwen3-Embedding-0.6B

# 4. 启动检索服务
python search_engine/faiss/retrieval_server.py --port 8080
```

## 项目结构

```
Domain-Specific-Deep-Research-agent/
├── src/
│   ├── embedding/              # Embedding 模型相关
│   │   ├── distill/            # Q&A 数据蒸馏
│   │   ├── train/              # InfoNCE 微调训练
│   │   └── eval/               # 检索评估
│   ├── generator/              # 生成模型相关
│   │   ├── distill/            # RAG 数据蒸馏
│   │   ├── sft/                # 监督微调
│   │   └── rl/                 # 强化学习训练
│   └── preprocess/             # 数据预处理
├── search_engine/
│   ├── faiss/                  # Faiss 向量检索
│   └── milvus/                 # Milvus 向量数据库
├── verl/                       # VeRL 强化学习框架
├── datasets/                   # 数据集目录
├── pretrain_models/            # 预训练权重
├── deploy/                     # 部署配置
├── docs/                       # 文档
│   └── tutorials.md            # 详细教程
└── requirements_retrieval.txt  # 检索系统依赖
```

## 核心模块

### 1. 检索系统

支持三种检索方式，可通过 API 调用：

```bash
# 启动检索服务
python search_engine/faiss/retrieval_server.py \
    --port 8080 \
    --retrieval_method hybrid \
    --alpha 0.6
```

| 检索方式 | 说明 |
|---------|------|
| `dense` | 基于 Qwen3-Embedding 的稠密向量检索 |
| `bm25` | 基于 Pyserini 的稀疏检索 |
| `hybrid` | 稠密 + 稀疏混合检索 |

### 2. Embedding 微调

使用 ms-swift 框架进行 InfoNCE 对比学习：

```bash
# 构建训练数据
python src/embedding/train/build_infonce_from_qa.py

# 开始训练
bash src/embedding/train/train_embedding.sh
```

### 3. 生成模型训练

#### SFT 监督微调

```bash
cd verl
bash custom/run_qwen_sft.sh
```

#### PRM 强化学习

```bash
# 启动奖励模型服务
bash verl/custom/reward_model/sglang_client.sh

# 开始 RL 训练
bash verl/custom/run_gsm8k_prm.sh
```

### 4. 模型部署

支持 GGUF 量化和 Ollama 部署：

```bash
# 转换为 GGUF 格式
python convert_hf_to_gguf.py ./pretrain_models/generator/Qwen3-8B/ \
    --outfile ./pretrain_models/generator/Qwen3_8B_Q4_K_M.gguf \
    --outtype q4_k_m

# Ollama 部署
ollama create Fin-Search -f ./deploy/ollama/Qwen3-8B_Q4_K_M/Modelfile
```

## 详细文档

完整的使用教程和技术细节请参考：

📖 **[docs/tutorials.md](docs/tutorials.md)**

包含以下内容：
- 环境配置与依赖安装
- 数据集清洗与预处理
- Q&A 两阶段蒸馏流程
- Embedding 微调训练
- 向量数据库构建
- RAG 评估数据蒸馏
- SFT 与 RL 训练
- 模型量化与部署
- 压力测试与能力评估

## 技术栈

| 组件 | 技术选型 |
|-----|---------|
| Embedding 模型 | Qwen3-Embedding-0.6B |
| 生成模型 | Qwen3-8B |
| 向量检索 | Faiss / Milvus |
| 稀疏检索 | Pyserini (BM25) |
| 训练框架 | VeRL / ms-swift |
| 推理加速 | SGLang |
| 部署方案 | Ollama / llama.cpp |

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [VeRL](https://github.com/volcengine/verl) - 强化学习训练框架
- [ms-swift](https://github.com/modelscope/swift) - Embedding 微调框架
- [SGLang](https://github.com/sgl-project/sglang) - 高效推理引擎
- [Qwen3](https://github.com/QwenLM/Qwen) - 基础模型
