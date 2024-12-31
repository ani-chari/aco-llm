from reasoning_graph import ReasoningGraph
from llm_agent import LLMAgent
from aco_optimizer import ACOOptimizer
from utils import (
    load_gsm8k, load_brain_teasers, load_brainteasers, load_math, 
    load_hellaswag, load_mmlu, load_mmlu_pro, load_alfworld, evaluate_path
)
import logging
import gc
from accelerate import Accelerator
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, filename='aco_results.log', filemode='w')

def main():
    try:
        # Initialize Accelerator for multi-GPU support
        accelerator = Accelerator()
        accelerator.print(f"Using {accelerator.num_processes} GPUs")

        # Load all datasets
        datasets = {
            'gsm8k': load_gsm8k(),
            'brain_teasers': load_brain_teasers(),
            'brainteasers': load_brainteasers(),
            'math': load_math(),
            'hellaswag': load_hellaswag(),
            'mmlu': load_mmlu(),
            'mmlu_pro': load_mmlu_pro(),
            'alfworld': load_alfworld()
        }

        # Initialize components
        reasoning_graph = ReasoningGraph()
        fine_tuned_models = [
            'qingy2024/QwQ-14B-Math-v0.2',
            'ank028/Llama-3.2-1B-Instruct-commonsense_qa',
            'hoskinson-center/proofGPT-v0.1-6.7B',
            'openGPT-X/Teuken-7B-instruct-research-v0.4'
        ]
        llm_agents = [LLMAgent(model_name) for model_name in fine_tuned_models]
        aco = ACOOptimizer(reasoning_graph)

        # Generate generalized trees of thought for each agent and dataset
        for dataset_name, (train_data, test_data) in datasets.items():
            if not train_data or not test_data:
                logging.warning(f"Failed to load {dataset_name} dataset. Skipping.")
                continue

            generalized_trees = []
            for agent in llm_agents:
                try:
                    # Generate trees of thought for a subset of problems
                    trees = []
                    for problem in train_data[:100]:  # Use a subset of 100 problems per agent
                        thought_steps, embeddings, validity_scores = agent.generate_tree_of_thought(problem['question'])
                        trees.append((thought_steps, embeddings, validity_scores))
                    # Consolidate the trees into a single generalized tree
                    generalized_steps, embeddings, validity_scores = agent.consolidate_trees(trees)
                    generalized_trees.append((generalized_steps, embeddings, validity_scores))
                    # Free up memory
                    del trees
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error generating generalized tree for agent on {dataset_name}: {e}")

            # Combine all generalized trees into the reasoning graph
            for generalized_steps, embeddings, validity_scores in generalized_trees:
                for step, emb, score in zip(generalized_steps, embeddings, validity_scores):
                    node_id = step
                    reasoning_graph.add_node(node_id, statement=step, embedding=emb, validity_score=score)
                    reasoning_graph.add_edge('start', node_id)
                    reasoning_graph.add_edge(node_id, 'end')

            # Merge similar nodes in the reasoning graph
            nodes_to_merge = []
            for u, v in reasoning_graph.graph.edges():
                if u != 'start' and v != 'end':
                    merged_node = reasoning_graph.merge_nodes(reasoning_graph.graph.nodes[u], reasoning_graph.graph.nodes[v])
                    if merged_node:
                        nodes_to_merge.append((u, v, merged_node))
            for u, v, merged_node in nodes_to_merge:
                reasoning_graph.graph.add_node(f'{u}|{v}', **merged_node)
                for predecessor in reasoning_graph.graph.predecessors(u):
                    reasoning_graph.graph.add_edge(predecessor, f'{u}|{v}', weight=reasoning_graph.graph[predecessor][u]['weight'])
                for successor in reasoning_graph.graph.successors(v):
                    reasoning_graph.graph.add_edge(f'{u}|{v}', successor, weight=reasoning_graph.graph[v][successor]['weight'])
                reasoning_graph.graph.remove_node(u)
                reasoning_graph.graph.remove_node(v)

            # ACO iterations
            best_path_quality = 0.0
            for iteration in range(100):  # Maximum of 100 iterations
                paths = []
                for agent in llm_agents:
                    try:
                        path = aco.traverse_graph(agent)
                        paths.append(path)
                    except Exception as e:
                        logging.error(f"Error traversing graph for agent on {dataset_name}: {e}")
                quality_scores = [evaluate_path(path, test_data, test_data, accelerator) for path in paths]  # Use full test set for evaluation
                aco.update_pheromones(paths, quality_scores)
                
                # Check for early stopping
                current_best_quality = max(quality_scores)
                if current_best_quality > best_path_quality:
                    best_path_quality = current_best_quality
                elif best_path_quality >= 0.95:  # Stop if the best path quality is 95% or higher
                    logging.info(f"Early stopping at iteration {iteration} (best path quality: {best_path_quality})")
                    break

            # Extract optimal chain of thought
            optimal_path = max([(path, evaluate_path(path, test_data, test_data, accelerator)) for path in paths], key=lambda x: x[1])[0]
            logging.info(f"Optimal Chain of Thought for {dataset_name}: {optimal_path}")

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")

if __name__ == '__main__':
    main()