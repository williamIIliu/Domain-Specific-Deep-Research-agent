import json
import os
import warnings
from typing import List, Dict, Optional, Union
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests


# ------------------------- 启动服务 -------------------------
if __name__ == "__main__":
    # 测试服务
    url = "http://localhost:8080/retrieve"

    data = {
        "queries": ["基金代码009730在2021年6月15日的单位净值是多少？"],
        "topk": 3,
        "return_scores": True,
        "retrieval_type": "dense",
        "hybrid_alpha": 0.5
    }

    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

    data = {
        "queries": ["基金代码009730在2021年6月15日的单位净值是多少？"],
        "topk": 3,
        "return_scores": True,
        "retrieval_type": "bm25",
        "hybrid_alpha": 0.5
    }

    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

    data = {
        "queries": ["基金代码009730在2021年6月15日的单位净值是多少？"],
        "topk": 3,
        "return_scores": True,
        "retrieval_type": "hybrid",
        "hybrid_alpha": 0.5
    }

    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

