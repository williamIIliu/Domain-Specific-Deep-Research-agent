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

以下是示例：
<Example1>
{{
  "Characters": "一位对个人理财与投资感兴趣的普通人士",
  "Passage": "基金代码007298，2019年8月27日交易日期对应的单位净值为1.0337，复权单位净值1.0337，累计单位净值1.0337，资产净值19797399.43元",
  "Question_Type": "acquire_knowledge",
  "Difficulty": "high_school"
}}
<Example2>
{{
  "Characters": "一位贸易金融专家，为国际贸易中的货币风险与支付风险缓解提供见解",
  "Passage": "2015年9月1日，工商银行安徽省分行与铜陵某上市公司签订LME铜标的远期合约，实现安徽省银行大宗商品衍生品交易零突破；该业务属上海自贸区可复制金融改革事项，外汇局安徽省分局推动政策落地，工行安徽分行此前已完成业务备案",
  "Question_Type": "background",
  "Difficulty": "university"
}}
<Example3>
{{
  "Characters": "一位研究行为金融的大学教授",
  "Passage": "基于2010-2023年A股个人投资者交易数据，本文验证了“损失厌恶系数与持仓周期呈负相关”的假设，且在创业板市场中该相关性比主板高17.3%（控制变量：投资者年龄、资金规模）",
  "Question_Type": "acquire_knowledge",
  "Difficulty": "phd"
}}

请基于用户提供的文档（Passage）和候选角色（Characters）生成输出结果。其中：
- 文档（Passage）的语言为中文， 为json格式为{{"id":" ","contents":" "}}，使用"contents"对应value进行判断；
- 候选角色（Characters）的描述语言为中文；
- 输出要求：仅返回 JSON 格式，且 JSON 中仅含"Characters","Question_Type"和"Difficulty"三个key及相应value，无需额外文本。

**Passage**: {passage}
**Characters**: {characters}
"""

emd_stage2 = """
给定一个**角色（Character）**、**文档（Passage）** 和**要求（Requirement）**，请从该角色的视角生成一条查询语句：需满足要求中的所有条件，且该查询能用于检索到指定的文档。最终结果仅以 JSON 格式返回，不包含任何额外文本。

## 格式规则
- **文档（Passage）** 语言：中文
- **角色（Character）** 与 **要求（Requirement）** 描述语言：中文
- **输出限制**：仅输出你认为合适的Generated_Query，而非json文件，无多余文本（如解释、说明、标点外的符号）

## 示例
<example>
{{
  "Character": "一位专攻期权交易的股票交易者，在金融论坛分享宝贵策略",
  "Passage": "2020年度63名激励对象可行权股票期权280.08万份，行权价格8.76元/股，应收款项合计2453.50万元",
  "Question_Type": "keywords",
  "Difficulty": "university",
}}
response："2020年度激励对象的股票期权行权价格是多少？"
</example>

## 输入参数
请基于以下参数生成查询：
- **Character**：{character}
- **Passage**：{passage}
- **Requirement**：
  1. 问题类型（Type）：{type}（在keywords/acquire_knowledge/summary/yes_or_no/background已经做出选择选择）
  2. 难度（Difficulty）：{difficulty}（需从high_school/university/phd中选择）
  3. 长度（Length）：根据角色身份、文档复杂度、问题类型和难度动态调整（关键词型偏短，总结型偏长；高难度问题可适当增加长度）
"""