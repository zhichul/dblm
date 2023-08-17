
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
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def save(self, file):
        with open(file, "w") as f:
            json.dump({
                    "nodes": [node.id for node in self._nodes],
                    "edges": [(edge.id, edge.nodes[0].id, edge.nodes[1].id) for edge in self._edges]
                    }, f)

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
