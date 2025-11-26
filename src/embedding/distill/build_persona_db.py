"""
构建角色画像的 Embedding 索引
使用 Qwen3-Embedding 模型和 FAISS 进行高效的相似性检索
"""
import json
import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """获取最后一个 token 的 hidden state 作为句子表示"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """构建指令格式的查询"""
    return f'Instruct: {task_description}\nQuery:{query}'


class PersonaIndexBuilder:
    """角色画像索引构建器"""
    
    def __init__(self, model_path: str = "./pretrain_models/embedding/qwen3-0.6b-embedding", 
                 max_length: int = 128, device: str = "auto"):
        """
        初始化索引构建器
        
        Args:
            model_path: Qwen3-Embedding 模型路径
            max_length: 最大序列长度
            device: 设备类型
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = device
        
        # 加载模型和分词器
        self._load_model()
        
        # 初始化 FAISS 索引
        self.index = None
        self.personas = []
        self.persona_embeddings = []
    
    def _load_model(self):
        """加载 Qwen3-Embedding 模型"""
        try:
            print(f"✅ 加载 Qwen3-Embedding 模型: {self.model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side='left',
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 设置设备
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.device)
            self.model.eval()
            
            # 获取 embedding 维度
            with torch.no_grad():
                # 创建一个测试输入来获取维度
                test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                test_input.to(self.device)
                outputs = self.model(**test_input)
                test_embedding = last_token_pool(outputs.last_hidden_state, test_input['attention_mask'])
                self.embedding_dim = test_embedding.shape[1]
            
            print(f"✅ 模型加载成功，embedding 维度: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def load_personas(self, persona_file: str) -> List[Dict[str, Any]]:
        """
        加载角色画像数据
        
        Args:
            persona_file: 角色画像文件路径
            
        Returns:
            角色画像列表
        """
        personas = []
        print(f"📖 加载角色画像文件: {persona_file}")
        
        with open(persona_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="读取角色画像")):
                try:
                    # 处理单引号格式的 JSON
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 替换单引号为双引号
                    json_str = line.replace("'", '"')
                    persona_data = json.loads(json_str)
                    
                    # 提取角色描述
                    persona_text = persona_data.get('persona', '')
                    if persona_text:
                        personas.append({
                            'id': i,
                            'persona': persona_text,
                            'original_data': persona_data
                        })
                        
                except Exception as e:
                    print(f"⚠️ 跳过无效行 {i+1}: {e}")
                    continue
        
        print(f"✅ 成功加载 {len(personas)} 个角色画像")
        return personas
    
    def generate_embeddings(self, personas: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        为角色画像生成 embeddings
        
        Args:
            personas: 角色画像列表
            batch_size: 批处理大小
            
        Returns:
            embeddings 数组
        """
        print(f"🔄 生成 {len(personas)} 个角色画像的 embeddings...")
        
        # 提取文本
        texts = [persona['persona'] for persona in personas]
        
        # 构建任务指令
        task = 'Given a persona description, find the most relevant personas for a given query'
        
        all_embeddings = []
        
        with torch.no_grad():
            # 批量处理
            for i in tqdm(range(0, len(texts), batch_size), desc="生成 embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # 构建指令格式的输入
                instructed_texts = [
                    get_detailed_instruct(task, text) for text in batch_texts
                ]
                
                # 分词
                batch_dict = self.tokenizer(
                    instructed_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict.to(self.device)
                
                # 获取 embeddings
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # 归一化
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # 合并所有 embeddings
        final_embeddings = np.vstack(all_embeddings)
        print(f"✅ 生成 embeddings 完成，形状: {final_embeddings.shape}")
        return final_embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        构建 FAISS 索引
        
        Args:
            embeddings: embeddings 数组
            
        Returns:
            FAISS 索引
        """
        print("🔨 构建 FAISS 索引...")
        
        # 创建 FAISS 索引 (使用 Inner Product，因为已经归一化了)
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 添加到索引
        index.add(embeddings.astype('float32'))
        
        print(f"✅ FAISS 索引构建完成，包含 {index.ntotal} 个向量")
        return index
    
    def save_index(self, index: faiss.Index, personas: List[Dict[str, Any]], 
                   embeddings: np.ndarray, output_dir: str = "./persona_index"):
        """
        保存索引和相关数据
        
        Args:
            index: FAISS 索引
            personas: 角色画像列表
            embeddings: embeddings 数组
            output_dir: 输出目录
        """
        print(f"💾 保存索引到: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 FAISS 索引
        index_path = os.path.join(output_dir, "persona_index.faiss")
        faiss.write_index(index, index_path)
        print(f"✅ FAISS 索引已保存: {index_path}")
        
        # 保存角色画像数据
        personas_path = os.path.join(output_dir, "personas.json")
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(personas, f, ensure_ascii=False, indent=2)
        print(f"✅ 角色画像已保存: {personas_path}")
        
        # 保存 embeddings (可选，用于调试)
        embeddings_path = os.path.join(output_dir, "persona_embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"✅ Embeddings 已保存: {embeddings_path}")
        
        # 保存元数据
        metadata = {
            "model_path": self.model_path,
            "embedding_dim": self.embedding_dim,
            "num_personas": len(personas),
            "index_type": "IndexFlatIP",
            "metric": "cosine_similarity",
            "max_length": self.max_length,
            "device": self.device
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ 元数据已保存: {metadata_path}")
        
        print(f"🎉 所有文件已保存到: {output_dir}")
    
    def build_index_from_file(self, persona_file: str, output_dir: str = "./persona_index"):
        """
        从文件构建完整的索引
        
        Args:
            persona_file: 角色画像文件路径
            output_dir: 输出目录
        """
        # 1. 加载角色画像
        personas = self.load_personas(persona_file)
        
        # 2. 生成 embeddings
        embeddings = self.generate_embeddings(personas)
        
        # 3. 构建 FAISS 索引
        index = self.build_faiss_index(embeddings)
        
        # 4. 保存所有数据
        self.save_index(index, personas, embeddings, output_dir)
        
        return index, personas, embeddings


class PersonaRetriever:
    """角色画像检索器"""
    
    def __init__(self, index_dir: str = "./datasets/persona_index"):
        """
        初始化检索器
        
        Args:
            index_dir: 索引目录
        """
        self.index_dir = index_dir
        self.load_index()
    
    def load_index(self):
        """加载索引和相关数据"""
        print(f"📂 从 {self.index_dir} 加载索引...")
        
        # 加载元数据
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 加载 FAISS 索引
        index_path = os.path.join(self.index_dir, "persona_index.faiss")
        self.index = faiss.read_index(index_path)
        
        # 加载角色画像
        personas_path = os.path.join(self.index_dir, "personas.json")
        with open(personas_path, 'r', encoding='utf-8') as f:
            self.personas = json.load(f)
        
        # 加载模型
        self._load_model()
        
        print(f"✅ 索引加载完成，包含 {len(self.personas)} 个角色画像")
    
    def _load_model(self):
        """加载模型"""
        try:
            model_path = self.metadata["model_path"]
            max_length = self.metadata.get("max_length", 128)
            device = self.metadata.get("device", "cpu")
            
            print(f"✅ 加载检索模型: {model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side='left',
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model.to(device)
            self.model.eval()
            
            self.max_length = max_length
            self.device = device
            
            print("✅ 检索模型加载成功")
            
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            raise e
    
    def retrieve_similar_personas(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相似的角色画像
        
        Args:
            query_text: 查询文本
            top_k: 返回前k个最相似的结果
            
        Returns:
            相似角色画像列表
        """
        # 生成查询 embedding
        query_embedding = self._generate_query_embedding(query_text)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # 有效索引
                persona = self.personas[idx].copy()
                persona['similarity_score'] = float(score)
                results.append(persona)
        
        return results
    
    def _generate_query_embedding(self, query_text: str) -> np.ndarray:
        """生成查询 embedding"""
        try:
            with torch.no_grad():
                # 构建任务指令
                task = 'Given a persona description, find the most relevant personas for a given query'
                instructed_query = get_detailed_instruct(task, query_text)
                
                # 分词
                batch_dict = self.tokenizer(
                    [instructed_query],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict.to(self.device)
                
                # 获取 embedding
                outputs = self.model(**batch_dict)
                embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # 归一化
                embedding = F.normalize(embedding, p=2, dim=1)
                
                return embedding.cpu().numpy()
                
        except Exception as e:
            print(f"⚠️ 生成查询 embedding 失败: {e}")
            # 返回零向量作为占位符
            return np.zeros((1, self.metadata.get("embedding_dim", 1024)))


def main():
    """主函数"""
    # 配置文件路径
    persona_file = "./datasets/persona-hub/finance_persona.jsonl"
    output_dir = "./datasets/persona-hub/finance_persona_index"
    
    # 构建索引 - 使用 Qwen3-Embedding
    builder = PersonaIndexBuilder(
        model_path="./pretrain_models/embedding/Qwen3-Embedding-0.6B",
        max_length=128,
        device="auto"
    )

    print("🚀 开始构建角色画像索引...")
    
    try:
        index, personas, embeddings = builder.build_index_from_file(
            persona_file=persona_file,
            output_dir=output_dir
        )
        
        print("\n🧪 测试检索功能...")
        
        # 测试检索
        retriever = PersonaRetriever(output_dir)
        
        # 测试查询
        test_queries = [
            "股票投资和基金理财",
            "小额信贷和创业支持", 
            "风险管理专家",
            "数据科学和机器学习"
        ]
        
        for query in test_queries:
            print(f"\n📝 查询: {query}")
            results = retriever.retrieve_similar_personas(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['similarity_score']:.3f}] {result['persona']}")
        
        print("\n✅ 索引构建和测试完成！")
        
    except Exception as e:
        print(f"❌ 构建索引失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()