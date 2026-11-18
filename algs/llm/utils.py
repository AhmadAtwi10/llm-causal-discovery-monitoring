import numpy as np


class PreviousEdges:
  def __init__(self):
      self.previous_edges = {}

  def add_node(self, node):
    if node not in self.previous_edges:
      self.previous_edges[node] = {
        "incoming":[],
        "outgoing":[]
      }
    
  def add_edge_pair_wise(self, head, tail, choice):
    self.add_node(head)
    self.add_node(tail)
    if choice == 'A':
      self.previous_edges[head]["outgoing"].append(tail)
      self.previous_edges[tail]["incoming"].append(head)
    elif choice == 'B':
      self.previous_edges[tail]["outgoing"].append(head)
      self.previous_edges[head]["incoming"].append(tail)
      
  def add_edge_full_graph(self, head, tails):
    self.add_node(head)
    for tail in tails:
      self.add_node(tail)
      self.previous_edges[head]["outgoing"].append(tail)
      self.previous_edges[tail]["incoming"].append(head)

  def get_previous_relevant_edges_string(self, head, tail):
    output_str = ''
    if head in self.previous_edges:
      for node in self.previous_edges[head]["outgoing"]:
        output_str += f'{head} causes {node}.\n'
      for node in self.previous_edges[head]["incoming"]:
        output_str += f'{node} causes {head}.\n'
      
    if tail in self.previous_edges:
      for node in self.previous_edges[tail]["outgoing"]:
        output_str += f'{tail} causes {node}.\n'
      for node in self.previous_edges[tail]["incoming"]:
        output_str += f'{node} causes {tail}.\n'
    return output_str
  
  def get_all_edges_string(self):
    output_str = ''
    for head in self.previous_edges:
      for node in self.previous_edges[head]["outgoing"]:
        output_str += f'{head} causes {node}.\n'
      for node in self.previous_edges[head]["incoming"]:
        output_str += f'{node} causes {head}.\n'
    return output_str
  
  def get_adjacency_matrix(self, vars):
    n = len(vars)
    adj_matrix = np.zeros((n, n))
    for i, var in enumerate(vars):
      if var in self.previous_edges:
        for node in self.previous_edges[var]["outgoing"]:
          j = vars.index(node)
          adj_matrix[i][j] = 1
    return adj_matrix
