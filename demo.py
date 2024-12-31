import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
from tqdm import tqdm

class ReasoningStep:
    def __init__(self, id: int, description: str, step_type: str):
        self.id = id
        self.description = description
        self.step_type = step_type

class BrainTeaserSolver:
    def __init__(self, num_agents: int = 5):
        self.agents = [Agent(f"Agent_{i}") for i in range(num_agents)]
        self.pheromone_matrix = {}
        self.steps = self.initialize_steps()
        self.best_path = None
        self.best_quality = 0
        self.training_history = []
        
    def initialize_steps(self) -> List[ReasoningStep]:
        steps_data = [
            ("Identify key objects/concepts in the riddle", "analysis"),
            ("List attributes and properties mentioned", "analysis"),
            ("Find contradictions or unusual combinations", "pattern"),
            ("Look for metaphorical meanings", "interpretation"),
            ("Consider common riddle patterns", "pattern"),
            ("Map attributes to potential objects", "mapping"),
            ("Test solution against all clues", "verification"),
            ("Check for wordplay or double meanings", "wordplay"),
            ("Consider literal vs figurative meanings", "interpretation"),
            ("Evaluate solution uniqueness", "verification")
        ]
        return [ReasoningStep(i, desc, type_) for i, (desc, type_) in enumerate(steps_data)]

    def visualize_reasoning_path(self, path: List[int], title: str):
        G = nx.DiGraph()
        
        # Create graph layout
        for step in self.steps:
            G.add_node(step.id, description=step.description, type=step.step_type)
        
        for i in range(len(path)-1):
            G.add_edge(path[i], path[i+1])
        
        # Custom circular layout
        pos = nx.circular_layout(G, scale=2)
        
        plt.figure(figsize=(15, 15))
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             width=2,
                             arrowsize=20,
                             arrowstyle='->',
                             connectionstyle='arc3,rad=0.2')
        
        # Draw nodes with custom style
        node_colors = ['lightblue' if n not in path else '#90EE90' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos,
                             node_color=node_colors,
                             node_size=3000,
                             alpha=0.7)
        
        # Add labels with better formatting
        labels = {node: self.wrap_text(G.nodes[node]['description'])
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels,
                              font_size=10,
                              font_weight='bold',
                              font_family='sans-serif')
        
        plt.title(title, pad=20, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()

    def wrap_text(self, text: str, width: int = 20) -> str:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word) + 1
        lines.append(' '.join(current_line))
        return '\n'.join(lines)

    def train(self, training_data: List[Dict], num_iterations: int = 10):
        print("\n=== Training Brain Teaser Solver ===")
        
        for iteration in tqdm(range(num_iterations)):
            iteration_qualities = []
            
            for problem_idx, problem in enumerate(training_data):
                print(f"\nProblem {problem_idx + 1}: {problem['teaser']}")
                
                for agent in self.agents:
                    path = agent.generate_path(self)
                    quality = self.evaluate_path(path, problem)
                    iteration_qualities.append(quality)
                    
                    if quality > self.best_quality:
                        self.best_quality = quality
                        self.best_path = path
                        print(f"\n★ New Best Path Found! Quality: {quality:.3f}")
                        self.visualize_reasoning_path(path, 
                            f"Current Best Reasoning Path (Quality: {quality:.3f})")
                        plt.show()
            
            avg_quality = np.mean(iteration_qualities)
            self.training_history.append(avg_quality)
            
            if iteration % 2 == 0:
                self.plot_training_progress()

    def evaluate_path(self, path: List[int], problem: Dict) -> float:
        # Simulate path quality evaluation
        path_length_score = 1.0 - (abs(len(path) - 6) / 10)
        step_diversity = len(set(path)) / len(path)
        type_diversity = len(set(self.steps[i].step_type for i in path)) / len(path)
        
        return (path_length_score + step_diversity + type_diversity) / 3

    def plot_training_progress(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history, 'b-', marker='o')
        plt.title("Training Progress", fontsize=14, pad=20)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Average Path Quality", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

class Agent:
    def __init__(self, name: str):
        self.name = name
        
    def generate_path(self, solver: BrainTeaserSolver) -> List[int]:
        available_steps = list(range(len(solver.steps)))
        path = [random.choice(available_steps)]
        available_steps.remove(path[0])
        
        while available_steps and len(path) < 6:
            next_step = self.select_next_step(path[-1], available_steps, solver)
            path.append(next_step)
            available_steps.remove(next_step)
        
        return path
    
    def select_next_step(self, current: int, available: List[int], 
                        solver: BrainTeaserSolver) -> int:
        if not available:
            return current
            
        probabilities = []
        for next_step in available:
            edge = (current, next_step)
            pheromone = solver.pheromone_matrix.get(edge, 0.1)
            heuristic = 1.0
            
            if solver.steps[next_step].step_type != solver.steps[current].step_type:
                heuristic = 1.2  # Favor different step types
                
            probability = pheromone * heuristic
            probabilities.append(probability)
        
        total = sum(probabilities)
        if total == 0:
            return random.choice(available)
        
        probabilities = [p/total for p in probabilities]
        return np.random.choice(available, p=probabilities)

def main():
    brain_teasers = [
        {
            'teaser': "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
            'solution': "An echo"
        },
        {
            'teaser': "The more you take, the more you leave behind. What am I?",
            'solution': "Footsteps"
        }
    ]

    solver = BrainTeaserSolver(num_agents=3)
    solver.train(brain_teasers, num_iterations=5)

    print("\n=== Final Results ===")
    print(f"Best path quality: {solver.best_quality:.3f}")
    print("\nOptimal reasoning path:")
    for i, step_id in enumerate(solver.best_path):
        step = solver.steps[step_id]
        print(f"{i+1}. {step.description} ({step.step_type})")

    solver.visualize_reasoning_path(solver.best_path, "Final Optimal Reasoning Path")
    plt.show()

if __name__ == "__main__":
    main()
