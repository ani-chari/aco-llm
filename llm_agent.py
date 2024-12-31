import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, pipeline

class LLMAgent:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.reasoning_pipeline = pipeline('text-generation', model=model_name, tokenizer=model_name)

    def generate_tree_of_thought(self, problem):
        # Generate thought steps
        prompt = f"Provide a detailed chain of thought to solve the following problem: {problem}"
        generated_text = self.reasoning_pipeline(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
        thought_steps = generated_text.split('\n')
        
        # Get embedding for the problem
        problem_inputs = self.tokenizer(problem, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            problem_outputs = self.model(**problem_inputs)
        problem_embedding = problem_outputs.last_hidden_state.mean(dim=1).numpy()
        problem_embedding = problem_embedding / np.linalg.norm(problem_embedding)
        
        # Get embeddings for thought steps
        thought_inputs = self.tokenizer(thought_steps, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            thought_outputs = self.model(**thought_inputs)
        thought_embeddings = thought_outputs.last_hidden_state.mean(dim=1).numpy()
        thought_embeddings = thought_embeddings / np.linalg.norm(thought_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities as validity_scores
        validity_scores = [np.dot(problem_embedding, step_embedding) for step_embedding in thought_embeddings]
        
        return thought_steps, thought_embeddings.tolist(), validity_scores

    def consolidate_trees(self, trees):
        # Collect all thought steps and their embeddings and validity_scores from trees
        all_thought_steps = []
        all_embeddings = []
        all_validity_scores = []
        for tree in trees:
            all_thought_steps.extend(tree[0])
            all_embeddings.extend(tree[1])
            all_validity_scores.extend(tree[2])
        
        # Convert all_embeddings to a numpy array for easier operations
        all_embeddings = np.array(all_embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        
        # Cluster similar thought steps based on cosine similarity
        similarity_threshold = 0.8
        clusters = []
        for i in range(len(all_embeddings)):
            added = False
            for cluster in clusters:
                # Compute similarity with the centroid of the cluster
                centroid = np.mean([all_embeddings[j] for j in cluster], axis=0)
                similarity = np.dot(all_embeddings[i], centroid)
                if similarity >= similarity_threshold:
                    cluster.append(i)
                    added = True
                    break
            if not added:
                clusters.append([i])
        
        # For each cluster, select a representative step, e.g., the one with the highest validity_score
        generalized_steps = []
        generalized_validity_scores = []
        for cluster in clusters:
            # Get the indices of thought_steps in this cluster
            indices = cluster
            # Get the validity_scores for these thought_steps
            cluster_validity_scores = [all_validity_scores[idx] for idx in indices]
            # Find the index of the highest validity_score in the cluster
            best_idx = indices[np.argmax(cluster_validity_scores)]
            # Select the corresponding thought_step and its validity_score
            generalized_steps.append(all_thought_steps[best_idx])
            generalized_validity_scores.append(all_validity_scores[best_idx])
        
        # Get embeddings for generalized_steps
        generalized_inputs = self.tokenizer(generalized_steps, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            generalized_outputs = self.model(**generalized_inputs)
        generalized_embeddings = generalized_outputs.last_hidden_state.mean(dim=1).numpy()
        generalized_embeddings = generalized_embeddings / np.linalg.norm(generalized_embeddings, axis=1, keepdims=True)
        
        return generalized_steps, generalized_embeddings.tolist(), generalized_validity_scores