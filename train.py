import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PeftModel,
    PeftConfig,
    default_data_collator,
    get_linear_schedule_with_warmup,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
)
from datasets import Dataset, DatasetDict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, required=True,
                        help='Task ID (1 to num-tasks)')
    parser.add_argument('--num-tasks', type=int, required=True,
                        help='Number of tasks')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the PeftModel directory')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to the input data file (in Arrow format)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum input length')
    parser.add_argument('--min-length', type=int, default=100,
                        help='Minimum output length')
    parser.add_argument('--num-beams', type=int, default=5,
                        help='Number of beams for beam search')
    parser.add_argument('--alpha', type=int, nargs='+',
                        default=[32, 64, 128, 256, 32, 64, 128, 256,
                                 32, 64, 128, 256, 16, 32, 64, 128, 256, 512],
                        help='Alpha values for LoRA')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log interval')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_file):
    pubmed_qa = Dataset.from_file(data_file)
    train_valid_test = pubmed_qa.train_test_split(test_size=0.55, shuffle=False)
    test_valid = train_valid_test['test'].train_test_split(test_size=0.909, shuffle=False)
    dataset_dict = DatasetDict({
        'train': train_valid_test['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })
    return dataset_dict

def preprocess_function(example):
    question = example['question']
    contexts = ''.join(example['context']['contexts'])
    answer = example['long_answer']
    return f'question: {question} context: {contexts} answer: {answer}'

def load_model(model_dir, alpha, device):
    config = PeftConfig.from_pretrained(model_dir)
    base_model_name_or_path = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side='left')
    model = Auto
