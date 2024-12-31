import networkx as nx
import numpy as np

class ReasoningGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.start_node = 'start'
        self.end_node = 'end'
        self.graph.add_node(self.start_node, statement='Start', embedding=np.zeros(768), validity_score=0.0)
        self.graph.add_node(self.end_node, statement='End', embedding=np.zeros(768), validity_score=0.0)

    def add_node(self, node_id, statement, embedding, validity_score):
        self.graph.add_node(node_id, statement=statement, embedding=embedding, validity_score=validity_score)

    def add_edge(self, u, v, weight=1.0):
        self.graph.add_edge(u, v, weight=weight)

    def merge_nodes(self, node1, node2, similarity_threshold=0.9):
        if self._cosine_similarity(node1['embedding'], node2['embedding']) > similarity_threshold:
            merged_statement = f"{node1['statement']} | {node2['statement']}"
            merged_embedding = (node1['embedding'] + node2['embedding']) / 2
            merged_validity = max(node1['validity_score'], node2['validity_score'])
            merged_node = {'statement': merged_statement, 'embedding': merged_embedding, 'validity_score': merged_validity}
            return merged_node
        else:
            return None

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))