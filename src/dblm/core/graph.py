from __future__ import annotations
from collections import defaultdict, deque
import json
import random

class Node:

    def __init__(self, i) -> None:
        self.id = i

    def __repr__(self):
        return f"Node({self.id})"

    def __str__(self):
        return self.__repr__()

class Factor:

    def __init__(self, i) -> None:
        self.id = i

    def __repr__(self):
        return f"Factor({self.id})"

    def __str__(self):
        return self.__repr__()

class Edge:

    def __init__(self, i, n1, n2) -> None:
        self.id = i
        self.nodes = (n1, n2)

    def __repr__(self):
        return f"Edge({self.nodes[0]}, {self.nodes[1]}, {self.id})"

    def __str__(self):
        return self.__repr__()

    @property
    def first_node_id(self):
        return self.nodes[0].id

    @property
    def second_node_id(self):
        return self.nodes[1].id

class FactorGraphEdge(Edge):
    @property
    def factor_id(self):
        return self.first_node_id

    @property
    def variable_id(self):
        return self.second_node_id
class Graph:
    """A graph object that should be gradually built up but not modified."""

    def __init__(self, n: int=0) -> None: # type: ignore
        # TODO: allow deleting an edge/node by nulling but not popping the edge/node
        self._nodes = []
        self._edges = []
        self._next_node_id = 0
        self._next_edge_id = 0
        for _ in range(n):
            self.add_node()

    def node2edge_index(self) -> dict[tuple[int, int], Edge]:
        index = {}
        for edge in self.edges:
            index[(edge.first_node_id, edge.second_node_id)] = edge
            index[(edge.second_node_id, edge.first_node_id)] = edge
        return index

    def add_node(self):
        node = Node(self._next_node_id)
        self._next_node_id += 1
        self._nodes.append(node)
        return node

    def add_edge(self, n1,  n2):
        if isinstance(n1, int):
            n1 = self._nodes[n1]
        if isinstance(n2, int):
            n2 = self._nodes[n2]
        edge = Edge(self._next_edge_id, n1, n2)
        self._next_edge_id += 1
        self._edges.append(edge)
        return edge

    @property
    def num_nodes(self):
        return len(self._nodes)

    @property
    def num_edges(self):
        return len(self._edges)

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    def save(self, file):
        with open(file, "w") as f:
            json.dump({
                    "nodes": [node.id for node in self._nodes],
                    "edges": [(edge.id, edge.nodes[0].id, edge.nodes[1].id) for edge in self._edges]
                    }, f)

    def save_json(self):
        return {
                    "nodes": [node.id for node in self._nodes],
                    "edges": [(edge.id, edge.nodes[0].id, edge.nodes[1].id) for edge in self._edges]
                    }
    @staticmethod
    def load_json(d):
        g = Graph()
        max_node_id = max(d["nodes"])
        max_edge_id = max(d["edges"], key=lambda x:x[0])[0]
        g._next_node_id = max_node_id + 1
        g._next_edge_id = max_edge_id + 1
        g._nodes = [None] * g._next_node_id
        for node_id in d["nodes"]:
            g._nodes[node_id] = Node(node_id)
        g._edges = [None] * g._next_edge_id
        for edge_id, child, parent in d["edges"]:
            g._edges[edge_id] = Edge(edge_id, g._nodes[child], g._nodes[parent])
        return g

    @staticmethod
    def load(file):
        with open(file) as f:
            d = json.load(f)
        g = Graph()
        max_node_id = max(d["nodes"])
        max_edge_id = max(d["edges"], key=lambda x:x[0])[0]
        g._next_node_id = max_node_id + 1
        g._next_edge_id = max_edge_id + 1
        g._nodes = [None] * g._next_node_id
        for node_id in d["nodes"]:
            g._nodes[node_id] = Node(node_id)
        g._edges = [None] * g._next_edge_id
        for edge_id, child, parent in d["edges"]:
            g._edges[edge_id] = Edge(edge_id, g._nodes[child], g._nodes[parent])
        return g

    def dfs(self, root=None, leaf_as_root=True):
        neighbors = defaultdict(list)
        for edge in self.edges:
            i, j = edge.first_node_id, edge.second_node_id
            neighbors[i].append(j)
            neighbors[j].append(i)
        if root is None:
            if leaf_as_root:
                nodes = list(range(self.num_nodes))
                random.shuffle(nodes)
                for src in nodes: 
                    if len(neighbors[src]) == 1:
                        root = src
                        break
                if root is None:raise AssertionError("No leaves in the graph.")
            else:
                root = 0
        visited = set()
        visited.add(root)
        order = []
        order.append((root, None))
        frontier = deque(((n, root) for n in neighbors[root]))
        while len(frontier) > 0:
            next, parent = frontier.popleft()
            visited.add(next)
            order.append((next, parent))
            for neighbor in neighbors[next]:
                if neighbor in visited: continue
                frontier.append((neighbor, next))
        if len(visited) != self.num_nodes:
            raise ValueError("Disconnected Graph")
        return order
class Chain(Graph):
    #TODO: make a directed chain with directed edges
    def __init__(self, n) -> None:
        super().__init__(n)
        for i in range(n-1):
            self.add_edge(i, i+1)

class FactorGraph(Graph):
    def __init__(self, nvars, nfactors) -> None:
        super().__init__()
        self._factors = []
        self._next_factor_id = 0
        for _ in range(nvars):
            self.add_node()
        for _ in range(nfactors):
            self.add_factor()

    def add_factor(self):
        factor = Factor(self._next_factor_id)
        self._next_factor_id += 1
        self._factors.append(factor)
        return factor

    def add_edge(self, factor,  node):
        if isinstance(factor, int):
            factor = self._factors[factor]
        if isinstance(node, int):
            node = self._nodes[node]
        if not (isinstance(factor, Factor) and isinstance(node, Node)):
            raise ValueError(f"FactorGraph add_edge expects (Factor, Node), got ({type(factor)}, {type(node)})")
        edge = FactorGraphEdge(self._next_edge_id, factor, node)
        self._next_edge_id += 1
        self._edges.append(edge)
        return edge

def random_labeled_tree(n: int):
    # generate a Prüfer Sequence
    pufer_seq = [random.randint(0, n-1) for _ in range(n-2)]
    # convert to tree
    graph = Graph(n)
    degrees = [1] * n
    for node in pufer_seq:
        degrees[node] = degrees[node] + 1
    for parent in pufer_seq:
        for children in range(n):
            if degrees[children] == 1:
                graph.add_edge(children, parent)
                degrees[children] = degrees[children] - 1
                degrees[parent] = degrees[parent] - 1
                break
    remaining_nodes = [i for i in range(n) if degrees[i] == 1]
    if len(remaining_nodes) != 2:
        raise AssertionError
    children, parent = remaining_nodes
    graph.add_edge(children, parent)
    degrees[children] = degrees[children] - 1
    degrees[parent] = degrees[parent] - 1
    if any(d != 0 for d in degrees):
        raise AssertionError
    return graph

if __name__ == "__main__":
    g = Graph(4)
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(0,3)
    for i in range(10):
        print(g.dfs(leaf_as_root=True))