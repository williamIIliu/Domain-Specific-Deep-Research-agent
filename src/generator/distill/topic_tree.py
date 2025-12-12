# topic_tree="""
# {"id": 0, "topic_name": "金融 - 无确切场景"}
# {"id": 1, "topic_name": "金融 - 银行业务 - 零售银行 - 储蓄账户"}
# {"id": 2, "topic_name": "金融 - 银行业务 - 零售银行 - 支票账户"}
# {"id": 3, "topic_name": "金融 - 银行业务 - 零售银行 - 信用卡"}
# {"id": 4, "topic_name": "金融 - 银行业务 - 零售银行 - 个人贷款"}
# {"id": 5, "topic_name": "金融 - 银行业务 - 商业银行 - 商业贷款"}
# {"id": 6, "topic_name": "金融 - 银行业务 - 商业银行 - 资产管理"}
# {"id": 7, "topic_name": "金融 - 银行业务 - 商业银行 - 贸易融资"}
# {"id": 8, "topic_name": "金融 - 银行业务 - 商业银行 - 企业账户"}
# {"id": 9, "topic_name": "金融 - 银行业务 - 投资银行 - 并购咨询"}
# {"id": 10, "topic_name": "金融 - 银行业务 - 投资银行 - 资本市场"}
# {"id": 11, "topic_name": "金融 - 银行业务 - 投资银行 - 财务顾问"}
# {"id": 12, "topic_name": "金融 - 银行业务 - 投资银行 - 风险管理"}
# {"id": 13, "topic_name": "金融 - 投资 - 股票市场 - 股票交易"}
# {"id": 14, "topic_name": "金融 - 投资 - 股票市场 - 股票分析"}
# {"id": 15, "topic_name": "金融 - 投资 - 股票市场 - IPO"}
# {"id": 16, "topic_name": "金融 - 投资 - 债券市场 - 政府债券"}
# {"id": 17, "topic_name": "金融 - 投资 - 债券市场 - 企业债券"}
# {"id": 18, "topic_name": "金融 - 投资 - 债券市场 - 债券交易"}
# {"id": 19, "topic_name": "金融 - 投资 - 基金 - 共同基金"}
# {"id": 20, "topic_name": "金融 - 投资 - 基金 - 交易所交易基金（ETF）"}
# {"id": 21, "topic_name": "金融 - 投资 - 基金 - 对冲基金"}
# {"id": 22, "topic_name": "金融 - 投资 - 衍生品市场 - 期货"}
# {"id": 23, "topic_name": "金融 - 投资 - 衍生品市场 - 期权"}
# {"id": 24, "topic_name": "金融 - 投资 - 衍生品市场 - 掉期"}
# """
# topic_tree_hash = ['金融 - 无确切场景', '金融 - 银行业务 - 零售银行 - 储蓄账户', '金融 - 银行业务 - 零售银行 - 支票账户', '金融 - 银行业务 - 零售银行 - 信用卡', '金融 - 银行业务 - 零售银行 - 个人贷款', '金融 - 银行业务 - 商业银行 - 商业贷款', '金融 - 银行业务 - 商业银行 - 资产管理', '金融 - 银行业务 - 商业银行 - 贸易融资', '金融 - 银行业务 - 商业银行 - 企业账户', '金融 - 银行业务 - 投资银行 - 并购咨询', '金融 - 银行业务 - 投资银行 - 资本市场', '金融 - 银行业务 - 投资银行 - 财务顾问', '金融 - 银行业务 - 投资银行 - 风险管理', '金融 - 投资 - 股票市场 - 股票交易', '金融 - 投资 - 股票市场 - 股票分析', '金融 - 投资 - 股票市场 - IPO', '金融 - 投资 - 债券市场 - 政府债券', '金融 - 投资 - 债券市场 - 企业债券', '金融 - 投资 - 债券市场 - 债券交易', '金融 - 投资 - 基金 - 共同基金', '金融 - 投资 - 基金 - 交易所交易基金（ETF）', '金融 - 投资 - 基金 - 对冲基金', '金融 - 投资 - 衍生品市场 - 期货', '金融 - 投资 - 衍生品市场 - 期权', '金融 - 投资 - 衍生品市场 - 掉期']

topic_tree = """
{"id": 0, "topic_name": "金融 - 银行业务 - 零售银行"}
{"id": 1, "topic_name": "金融 - 银行业务 - 商业银行"}
{"id": 2, "topic_name": "金融 - 银行业务 - 投资银行"}
{"id": 3, "topic_name": "金融 - 投资 - 股票市场"}
{"id": 4, "topic_name": "金融 - 投资 - 债券市场"}
{"id": 5, "topic_name": "金融 - 投资 - 基金"}
{"id": 6, "topic_name": "金融 - 投资 - 衍生品市场"}
{"id": 7, "topic_name": "金融 - 保险 - 人寿保险"}
{"id": 8, "topic_name": "金融 - 保险 - 财产保险"}
{"id": 9, "topic_name": "金融 - 保险 - 健康保险"}
{"id": 10, "topic_name": "金融 - 金融科技 - 区块链"}
{"id": 11, "topic_name": "金融 - 金融科技 - 人工智能"}
{"id": 12, "topic_name": "金融 - 金融科技 - 大数据"}
{"id": 13, "topic_name": "金融 - 监管与合规 - 反洗钱（AML）"}
{"id": 14, "topic_name": "金融 - 监管与合规 - 合规审计"}
{"id": 15, "topic_name": "金融 - 监管与合规 - 监管报告"}
"""
topic_tree_hash = ["金融 - 银行业务 - 零售银行","金融 - 银行业务 - 商业银行","金融 - 银行业务 - 投资银行","金融 - 投资 - 股票市场","金融 - 投资 - 债券市场","金融 - 投资 - 基金","金融 - 投资 - 衍生品市场","金融 - 保险 - 人寿保险","金融 - 保险 - 财产保险","金融 - 保险 - 健康保险","金融 - 金融科技 - 区块链","金融 - 金融科技 - 人工智能","金融 - 金融科技 - 大数据","金融 - 监管与合规 - 反洗钱（AML）","金融 - 监管与合规 - 合规审计","金融 - 监管与合规 - 监管报告"]

# 中文主题名到英文目录名的映射
topic_name_zh_to_en = {
    # 一级分类
    "金融": "finance",
    
    # 银行业务
    "银行业务": "banking",
    "零售银行": "retail_banking",
    "商业银行": "commercial_banking",
    "投资银行": "investment_banking",
    "储蓄账户": "savings_account",
    "支票账户": "checking_account",
    "信用卡": "credit_card",
    "个人贷款": "personal_loan",
    "商业贷款": "commercial_loan",
    "资产管理": "asset_management",
    "贸易融资": "trade_finance",
    "企业账户": "corporate_account",
    "并购咨询": "ma_advisory",
    "资本市场": "capital_markets",
    "财务顾问": "financial_advisory",
    "风险管理": "risk_management",
    
    # 投资
    "投资": "investment",
    "股票市场": "stock_market",
    "股票交易": "stock_trading",
    "股票分析": "stock_analysis",
    "IPO": "ipo",
    "债券市场": "bond_market",
    "政府债券": "government_bonds",
    "企业债券": "corporate_bonds",
    "债券交易": "bond_trading",
    "基金": "funds",
    "共同基金": "mutual_funds",
    "交易所交易基金（ETF）": "etf",
    "对冲基金": "hedge_funds",
    "衍生品市场": "derivatives_market",
    "期货": "futures",
    "期权": "options",
    "掉期": "swaps",
    
    # 保险
    "保险": "insurance",
    "人寿保险": "life_insurance",
    "财产保险": "property_insurance",
    "健康保险": "health_insurance",
    
    # 金融科技
    "金融科技": "fintech",
    "区块链": "blockchain",
    "人工智能": "artificial_intelligence",
    "大数据": "big_data",
    
    # 监管与合规
    "监管与合规": "regulation_compliance",
    "反洗钱（AML）": "aml",
    "合规审计": "compliance_audit",
    "监管报告": "regulatory_reporting",
    
    # 其他
    "无确切场景": "general",
}


def translate_topic_path(topic_name: str) -> str:
    """
    将中文主题路径转换为英文目录路径
    
    Args:
        topic_name: 中文主题名称，如 "金融 - 投资 - 股票市场"
    
    Returns:
        英文目录路径，如 "finance/investment/stock_market"
    """
    if not topic_name:
        return "unknown"
    
    # 按 " - " 分割主题层级
    parts = topic_name.split(" - ")
    
    # 翻译每个部分
    translated_parts = []
    for part in parts:
        part = part.strip()
        if part in topic_name_zh_to_en:
            translated_parts.append(topic_name_zh_to_en[part])
        else:
            # 如果没有映射，使用拼音或保持原样（这里简单处理为下划线连接）
            translated_parts.append(part.replace(" ", "_"))
    
    return "/".join(translated_parts)