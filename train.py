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
    parser.add_argument('--model-name', type=str, required=True,
                        help='Pretrained model name or path')
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
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
    
def load_model(model_dir, alpha, device):
    config = PeftConfig.from_pretrained(model_dir)
    base_model_name_or_path = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model = PeftModel(model, config, alpha=alpha)
    model.to(device)
    return model, tokenizer

def train(model, tokenizer, dataset_dict, args):
    train_dataset = dataset_dict['train'].map(preprocess_function, batched=True, remove_columns=['question', 'context', 'long_answer'])
    eval_dataset = dataset_dict['valid'].map(preprocess_function, batched=True, remove_columns=['question', 'context', 'long_answer'])

    train_dataset = train_dataset.map(lambda x: tokenizer(x['input_ids'], padding='max_length', truncation=True, max_length=args.max_length), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenizer(x['input_ids'], padding='max_length', truncation=True, max_length=args.max_length), batched=True)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=default_data_collator)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=1000,
    )
    model.zero_grad()
    set_seed(args.seed)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss / args.batch_size
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            model.zero_grad()
            if step % args.log_interval == 0:
                logger.info(f'Epoch {epoch}, Step {step}, Loss {loss.item():.4f}')
                logger.info('Generating examples:')
                generate_examples(model, tokenizer, eval_dataloader, args)

def generate_examples(model, tokenizer, dataloader, args):
    model.eval()
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=args.device)
    logits_processor = LogitsProcessorList([
        TemperatureLogitsWarper(temperature=1.0),
        RepetitionPenaltyLogitsProcessor(1.2)
    ])
    for batch in dataloader:
        input_ids = batch['input_ids'].to(args.device)
        generated_text = generator(input_ids=input_ids,
                                   max_length=args.max_length,
                                   min_length=args.min_length,
                                   num_beams=args.num_beams,
                                   logits_processor=logits_processor,
                                   return_tensors='pt')
        generated_text = tokenizer.batch_decode(generated_text['input_ids'], skip_special_tokens=True)
        logger.info(generated_text)

if __name__ == '__main__':
    args = parse_args()
    dataset_dict = load_data(args.data_file)
    model, tokenizer = load_model(args.model_dir, args.alpha, args.device)
    train(model, tokenizer, dataset_dict, args)

