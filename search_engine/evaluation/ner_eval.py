import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parse_ner_json(json_str):
    """
    Parse the NER JSON output.
    Returns:
        dict: The parsed entities.
        bool: Whether the JSON was valid.
    """
    try:
        # Find the first { and last } to handle potential noise
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
            
        data = json.loads(json_str)
        # Normalize: ensure keys exist and values are lists
        normalized = {}
        target_keys = ["ORG", "PRODUCT", "METRIC", "TERM", "TIME"]
        for k in target_keys:
            val = data.get(k, [])
            if isinstance(val, list):
                # Ensure all items are strings
                normalized[k] = [str(item) for item in val]
            else:
                normalized[k] = []
        return normalized, True
    except:
        return {"ORG": [], "PRODUCT": [], "METRIC": [], "TERM": [], "TIME": []}, False

def extract_entities(ner_dict):
    """
    Flatten dictionary to a set of (category, entity) tuples.
    """
    entities = set()
    for label, items in ner_dict.items():
        for item in items:
            entities.add((label, item))
    return entities

def evaluate_local(model_path, data_path, output_file=None):
    """Evaluate using local model (transformers)"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
    trust_remote_code=True,
    # fix_mistral_regex=True  # 核心修复参数
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total = 0
    json_valid_count = 0
    
    # Global metrics counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    results = []
    
    print("Starting evaluation...")
    for line in tqdm(lines):
        try:
            item = json.loads(line)
        except:
            continue
            
        messages = item.get('messages', [])
        if not messages:
            continue
            
        # Extract ground truth from the last assistant message
        if messages[-1]['role'] == 'assistant':
            target_text = messages[-1]['content']
            input_messages = messages[:-1]
        else:
            # Skip if no ground truth
            continue
            
        # Apply chat template
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy decoding
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse prediction and ground truth
        pred_dict, is_valid = parse_ner_json(response_text)
        gt_dict, _ = parse_ner_json(target_text)
        
        if is_valid:
            json_valid_count += 1
            
        pred_entities = extract_entities(pred_dict)
        gt_entities = extract_entities(gt_dict)
        
        common = pred_entities.intersection(gt_entities)
        
        tp = len(common)
        fp = len(pred_entities) - tp
        fn = len(gt_entities) - tp
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
        total += 1
        
        results.append({
            "target": target_text,
            "prediction": response_text,
            "is_valid_json": is_valid,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

    # Calculate metrics
    json_acc = json_valid_count / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"Total Samples: {total}")
    print(f"JSON Output Accuracy: {json_acc:.4f}")
    print(f"Overall Entity Precision: {precision:.4f}")
    print(f"Overall Entity Recall: {recall:.4f}")
    print(f"Overall Entity F1: {f1:.4f}")
    print("="*30)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(f"Detailed results saved to {output_file}")

def evaluate_vllm(api_base, data_path, output_file=None, model_name="qwen-ner", num_workers=16):
    """Evaluate using vLLM deployed model via OpenAI API with parallel requests"""
    print(f"Connecting to vLLM server at {api_base}...")
    client = OpenAI(
        base_url=f"{api_base}/v1",
        api_key="EMPTY"  # vLLM doesn't require API key
    )
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        "pretrain_models/ner/Qwen3-4B-Base",
        trust_remote_code=True
    )
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Prepare all samples first
    samples = []
    for line in lines:
        try:
            item = json.loads(line)
        except:
            continue
            
        messages = item.get('messages', [])
        if not messages:
            continue
            
        # Extract ground truth from the last assistant message
        if messages[-1]['role'] == 'assistant':
            target_text = messages[-1]['content']
            input_messages = messages[:-1]
        else:
            continue
            
        # Apply chat template
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        samples.append({
            "text": text,
            "target_text": target_text
        })
    
    print(f"Total samples: {len(samples)}")
    print(f"Using {num_workers} parallel workers")
    
    # Thread-safe results storage
    results_lock = threading.Lock()
    results = []
    
    def process_sample(sample, idx):
        """Process a single sample"""
        try:
            response = client.completions.create(
                model=model_name,
                prompt=sample["text"],
                max_tokens=512,
                temperature=0.0,
            )
            response_text = response.choices[0].text
        except Exception as e:
            print(f"API call failed: {e}")
            response_text = ""
        
        # Parse prediction and ground truth
        pred_dict, is_valid = parse_ner_json(response_text)
        gt_dict, _ = parse_ner_json(sample["target_text"])
        
        pred_entities = extract_entities(pred_dict)
        gt_entities = extract_entities(gt_dict)
        
        common = pred_entities.intersection(gt_entities)
        
        tp = len(common)
        fp = len(pred_entities) - tp
        fn = len(gt_entities) - tp
        
        return {
            "idx": idx,
            "target": sample["target_text"],
            "prediction": response_text,
            "is_valid_json": is_valid,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    # Parallel processing
    print("Starting parallel evaluation...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, sample, i): i for i, sample in enumerate(samples)}
        
        for future in tqdm(as_completed(futures), total=len(samples)):
            result = future.result()
            with results_lock:
                results.append(result)
    
    # Reorder results by original index
    results = sorted(results, key=lambda x: x["idx"])
    
    # Calculate metrics
    total = len(results)
    json_valid_count = sum(1 for r in results if r["is_valid_json"])
    true_positives = sum(r["tp"] for r in results)
    false_positives = sum(r["fp"] for r in results)
    false_negatives = sum(r["fn"] for r in results)
    
    json_acc = json_valid_count / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"Total Samples: {total}")
    print(f"JSON Output Accuracy: {json_acc:.4f}")
    print(f"Overall Entity Precision: {precision:.4f}")
    print(f"Overall Entity Recall: {recall:.4f}")
    print(f"Overall Entity F1: {f1:.4f}")
    print("="*30)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(f"Detailed results saved to {output_file}")

def evaluate(model_path=None, data_path=None, output_file=None, 
             use_vllm=False, api_base="http://localhost:8000", model_name="qwen-ner", num_workers=16):
    """Main evaluation function - choose between local or vLLM"""
    if use_vllm:
        evaluate_vllm(api_base, data_path, output_file, model_name, num_workers)
    else:
        evaluate_local(model_path, data_path, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--model_path", type=str, default="output/ner/Qwen3-4B-final", help="Path to local model")
    parser.add_argument("--data_path", type=str, default="datasets/NER/FiNER_Eval.jsonl", help="Path to evaluation data")
    parser.add_argument("--output_file", type=str, default="datasets/NER/test_results.jsonl", help="Output file")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM API instead of local model")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000", help="vLLM API base URL")
    parser.add_argument("--model_name", type=str, default="qwen-ner", help="Model name in vLLM")
    parser.add_argument("--num_workers", type=int, default=128, help="Number of parallel workers for vLLM API calls")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        output_file=args.output_file,
        use_vllm=args.use_vllm,
        api_base=args.api_base,
        model_name=args.model_name,
        num_workers=args.num_workers
    )
