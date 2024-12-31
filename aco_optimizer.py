import numpy as np

class ACOOptimizer:
    def __init__(self, graph, num_agents=10, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        self.graph = graph
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

    def traverse_graph(self, agent):
        current_node = 'start'
        path = [current_node]
        while current_node != 'end':
            successors = list(self.graph.graph.successors(current_node))
            if not successors:
                break
            probabilities = self._calculate_probabilities(current_node, successors, agent)
            chosen_node = np.random.choice(successors, p=probabilities)
            path.append(chosen_node)
            current_node = chosen_node
        return path

    def _calculate_probabilities(self, current_node, successors, agent):
        pheromones = [self.graph.graph[current_node][succ]['weight'] for succ in successors]
        heuristics = [agent.get_heuristic_evaluation(self.graph.graph.nodes[succ]) for succ in successors]
        numerator = [ (pheromone ** self.alpha) * (heuristic ** self.beta) for pheromone, heuristic in zip(pheromones, heuristics)]
        probabilities = numerator / np.sum(numerator)
        return probabilities

    def update_pheromones(self, paths, quality_scores):
        for edge in self.graph.graph.edges():
            self.graph.graph[edge[0]][edge[1]]['weight'] *= (1 - self.evaporation_rate)
        for path, score in zip(paths, quality_scores):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if u in self.graph.graph and v in self.graph.graph[u]:
                    self.graph.graph[u][v]['weight'] += score