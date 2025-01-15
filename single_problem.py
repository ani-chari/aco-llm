# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import networkx as nx
import numpy as np
import random

# Set up HuggingFace access (replace with actual token if needed)
# from huggingface_hub import HuggingFaceHubAuthentication
# HuggingFaceHubAuthentication.set_access_token('YOUR_ACCESS_TOKEN')

# Define model names
MAIN_LLM_NAME = 'meta-llama/Llama-3.3-70B-Instruct'

# Load main LLM and tokenizer
main_llm = AutoModelForCausalLM.from_pretrained(MAIN_LLM_NAME)
main_tokenizer = AutoTokenizer.from_pretrained(MAIN_LLM_NAME)

# Load ant models and tokenizers (assuming similar architecture)
ANT_MODELS = [
    'qingy2024/QwQ-14B-Math-v0.2',
    'ank028/Llama-3.2-1B-Instruct-commonsense_qa',
    'hoskinson-center/proofGPT-v0.1-6.7B',
    'openGPT-X/Teuken-7B-instruct-research-v0.4'
]

ant_models = [AutoModelForCausalLM.from_pretrained(name) for name in ANT_MODELS]
ant_tokenizers = [AutoTokenizer.from_pretrained(name) for name in ANT_MODELS]

def generate_reasoning_graph(main_llm, main_tokenizer, problem, max_depth=3):
    graph = nx.DiGraph()
    graph.add_node('start')
    graph.add_node('end')
    
    current_nodes = ['start']
    depth = 0
    
    while current_nodes and depth < max_depth:
        next_nodes = []
        for node in current_nodes:
            prompt = f"From {node}, think of possible steps to solve: {problem}"
            inputs = main_tokenizer.encode(prompt, return_tensors='pt')
            outputs = main_llm.generate(inputs, max_length=200)
            thoughts = main_tokenizer.decode(outputs[0], skip_special_tokens=True).split('.')
            
            for i, thought in enumerate(thoughts):
                new_node = f"{node}_thought_{i}"
                graph.add_node(new_node)
                graph.add_edge(node, new_node)
                next_nodes.append(new_node)
            
            graph.add_edge(node, 'end')
        
        current_nodes = next_nodes
        depth += 1
    
    return graph

def ant_traverse(graph, ant_model, ant_tokenizer, pheromones, alpha=1, beta=1):
    current_node = 'start'
    path = ['start']
    while current_node != 'end':
        neighbors = list(graph.successors(current_node))
        if not neighbors:
            break
        # Calculate probabilities based on pheromones and heuristic
        denom = sum((pheromones[(current_node, n)] ** alpha) * (1 / (1 + len(n))) ** beta for n in neighbors)
        probs = [(pheromones[(current_node, n)] ** alpha) * (1 / (1 + len(n))) ** beta / denom for n in neighbors]
        next_node = random.choices(neighbors, weights=probs)[0]
        path.append(next_node)
        current_node = next_node
    return path

def update_pheromones(graph, paths, pheromones, rho=0.5, Q=1):
    # Evaporate pheromones
    for u, v in pheromones:
        pheromones[(u, v)] *= (1 - rho)
    # Add pheromones based on path quality
    for path in paths:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            pheromones[(u, v)] += Q / len(path)

def evaluate_path(graph, path, embeddings):
    # Coherence score
    coherence = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        coherence += np.dot(embeddings[u], embeddings[v]) / (np.norm(embeddings[u]) * np.norm(embeddings[v]))
    coherence /= len(path) - 1
    
    # Path length penalty
    length_penalty = -len(path)
    
    # Quality vote from ant LLMs
    quality_votes = []
    for model, tokenizer in zip(ant_models, ant_tokenizers):
        # Simplified quality assessment
        inputs = tokenizer.encode(f"Is this path good? {path}", return_tensors='pt')
        outputs = model.generate(inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "yes" in response.lower():
            quality_votes.append(1)
        else:
            quality_votes.append(0)
    quality_score = np.mean(quality_votes)
    
    # Overall score
    overall_score = coherence + length_penalty + quality_score
    return overall_score

def optimize_chain_of_thought(main_llm, main_tokenizer, ant_models, ant_tokenizers, problem, iterations=10, max_depth=3):
    graph = generate_reasoning_graph(main_llm, main_tokenizer, problem, max_depth)
    pheromones = {(u, v): 1 for u, v in graph.edges()}
    
    for _ in range(iterations):
        # Ant traversal
        paths = [ant_traverse(graph, model, tokenizer, pheromones) for model, tokenizer in zip(ant_models, ant_tokenizers)]
        
        # Evaluate paths
        path_scores = [evaluate_path(graph, path, ant_models, ant_tokenizers) for path in paths]
        
        # Update pheromones
        update_pheromones(graph, paths, pheromones)
    
    # Find the best path
    all_paths = list(nx.all_simple_paths(graph, 'start', 'end'))
    best_path = max(all_paths, key=lambda path: evaluate_path(graph, path, ant_models, ant_tokenizers))
    return best_path

def generate_final_answer(main_llm, main_tokenizer, best_path, problem):
    thoughts = ' -> '.join(best_path[1:-1])
    prompt = f"Use the following chain of thought to solve the problem: {thoughts}. Problem: {problem}"
    inputs = main_tokenizer.encode(prompt, return_tensors='pt')
    outputs = main_llm.generate(inputs, max_length=500)
    answer = main_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def validate_answer(main_llm, main_tokenizer, answer):
    prompt = f"Is this answer correct? {answer}"
    inputs = main_tokenizer.encode(prompt, return_tensors='pt')
    outputs = main_llm.generate(inputs)
    response = main_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "yes" in response.lower():
        return True
    else:
        return False
    
# Load MATH dataset
dataset = load_dataset('lighteval/MATH')

# Select a problem from the dataset
problem = dataset['train'][0]['question']

# Optimize chain of thought
best_path = optimize_chain_of_thought(main_llm, main_tokenizer, ant_models, ant_tokenizers, problem, iterations=5, max_depth=3)

# Generate final answer
final_answer = generate_final_answer(main_llm, main_tokenizer, best_path, problem)

# Validate answer
is_correct = validate_answer(main_llm, main_tokenizer, final_answer)

print(f"Final Answer: {final_answer}")
print(f"Is the answer correct? {is_correct}")