import pandas as pd
import json
import os
import numpy as np

def parse_jsonl_to_list(file_path, source_name):
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found.")
        return data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                messages = item.get('messages', [])
                
                # Extract system, prompt (user) and response (assistant)
                system_prompt = ""
                user_prompt = ""
                assistant_response = ""
                
                for msg in messages:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    elif msg['role'] == 'user':
                        user_prompt = msg['content']
                    elif msg['role'] == 'assistant':
                        assistant_response = msg['content']
                
                if not user_prompt or not assistant_response:
                    continue
                
                # Combine system and user prompt as the final prompt for training
                # This follows the ChatML/Instruct style
                full_prompt = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt
                
                data.append({
                    'id': f"{source_name}_{i}",
                    'prompt': full_prompt,
                    'response': assistant_response,
                    'mask': True,
                    'length': len(full_prompt),
                    'source': source_name
                })
            except Exception as e:
                print(f"Error processing {file_path} line {i}: {e}")
                continue
    return data

def process_finer_to_parquet(train_jsonl, test_jsonl, output_dir):
    print(f"📖 Processing train data: {train_jsonl}")
    train_data = parse_jsonl_to_list(train_jsonl, 'FiNER_train')
    
    print(f"📖 Processing test data: {test_jsonl}")
    test_data = parse_jsonl_to_list(test_jsonl, 'FiNER_test')
    
    all_data = train_data + test_data
    df_all = pd.DataFrame(all_data)
    
    # Calculate statistics (prompt + response)
    df_all['total_length'] = df_all['prompt'].str.len() + df_all['response'].str.len()
    avg_len = df_all['total_length'].mean()
    max_len = df_all['total_length'].max()
    
    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, 'train.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"✅ Data processing complete.")
    print(f"📍 Train path: {train_path} ({len(train_df)} samples)")
    print(f"📍 Test path: {test_path} ({len(test_df)} samples)")
    print(f"📊 Average length (prompt+response): {avg_len:.2f}")
    print(f"📊 Max length (prompt+response): {max_len}")
    
    return avg_len, max_len

if __name__ == "__main__":
    train_file = "/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/NER/FiNER.jsonl"
    test_file = "/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/NER/FiNER_Eval.jsonl"
    output_directory = "/mypool/lzq/LLM/Domain-Specific-Deep-Research-agent/datasets/NER"
    process_finer_to_parquet(train_file, test_file, output_directory)
