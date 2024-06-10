import pickle
import fire
from typing import List, Tuple, Dict
import json
from sentence_transformers import SentenceTransformer
import os
import glob
from tqdm import tqdm
import torch


def postprocess_response(response: str, strategy: str) -> str:
    # 将响应转换为小写并去除前后的空格
    seq_txt = response.lower().strip()

    # 去除响应中的 "assistant:" 前缀
    # 取出按照"assistant:"分割后的最后一部分
    if 'assistant:' in seq_txt:
        seq_txt = seq_txt.split('assistant:')[-1]
    # 删除确切的策略提及
    # 首先将字符串转换为小写，然后再进行替换，这样可以确保无论原始文本中的词语是大写还是小写，都可以被正确地替换
    seq_txt = seq_txt.replace(strategy.lower(), '')

    return seq_txt.strip()


def load_and_postprocess_data(response_file: str) -> Dict:
    with open(response_file, 'r') as f:
        data = json.load(f)

    responses = data['responses']

    for strategy, response in responses.items():
        data['responses'][strategy] = postprocess_response(response, strategy)

    return data


def run(data_dir='outputs/exp3_70b_c1_hf_partial', output_path='../data/output.json'):
    result = []

    print("retrieving and postprocessing responses...")
    paths = glob.glob(os.path.join(data_dir, '*.json'))
    for file in tqdm(paths):
        data = load_and_postprocess_data(file)
        result.append(data)

    print(f"total conversations: {len(result)}")
    print(f"total continuations: {sum([len(data['responses']) for data in result])}")

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    fire.Fire(run)
