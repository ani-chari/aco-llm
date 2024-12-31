from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import logging
import torch
from accelerate import Accelerator

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_gsm8k():
    try:
        # Load the full GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main")
        return dataset['train'], dataset['test']  # Return train and test splits
    except Exception as e:
        logging.error(f"Error loading GSM8K dataset: {e}")
        return [], []

def load_brain_teasers():
    try:
        dataset = load_dataset("ErfanMoosaviMonazzah/brain-teasers")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading Brain Teasers dataset: {e}")
        return [], []

def load_brainteasers():
    try:
        dataset = load_dataset("tasksource/brainteasers")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading Brainteasers dataset: {e}")
        return [], []

def load_math():
    try:
        dataset = load_dataset("lighteval/MATH")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading MATH dataset: {e}")
        return [], []

def load_hellaswag():
    try:
        dataset = load_dataset("Rowan/hellaswag")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading HellaSwag dataset: {e}")
        return [], []

def load_mmlu():
    try:
        dataset = load_dataset("cais/mmlu")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading MMLU dataset: {e}")
        return [], []

def load_mmlu_pro():
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading MMLU-Pro dataset: {e}")
        return [], []

def load_alfworld():
    try:
        dataset = load_dataset("alfworld")
        return dataset['train'], dataset['test']
    except Exception as e:
        logging.error(f"Error loading AlfWorld dataset: {e}")
        return [], []

def evaluate_path(path, problems, solutions, accelerator):
    try:
        # Use a smaller model for evaluation to reduce memory usage
        llm = AutoModel.from_pretrained('meta-llama/Llama-3.3-7B-Instruct', device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-7B-Instruct')
        llm, tokenizer = accelerator.prepare(llm, tokenizer)
        correct_count = 0
        for problem, solution in zip(problems, solutions):
            prompt = f"Use the following chain of thought to solve the problem: {path}\nProblem: {problem['question']}"
            inputs = tokenizer(prompt, return_tensors='pt').to(accelerator.device)
            outputs = llm.generate(**inputs, max_length=100)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Use the same model to grade correctness
            grade_prompt = f"Grade the correctness of the following answer: {answer}\nCorrect answer: {solution['answer']}"
            grade_inputs = tokenizer(grade_prompt, return_tensors='pt').to(accelerator.device)
            grade_outputs = llm.generate(**grade_inputs, max_length=100)
            grade = tokenizer.decode(grade_outputs[0], skip_special_tokens=True)
            if 'correct' in grade.lower():
                correct_count += 1
        return correct_count / len(problems)
    except Exception as e:
        logging.error(f"Error evaluating path: {e}")
        return 0.0