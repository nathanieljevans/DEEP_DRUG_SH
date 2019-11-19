import pandas as pd 
import sys 
sys.setrecursionlimit(1000)

class explorer: 
  def __init__(self, graph, node): 
    self.graph = graph
    self.explored = set([node])
    self.traverse(node)
    
  def traverse(self, node): 
    print(f'n explored = {len(self.explored)}', end='\r')
    
    to_explore = self.graph[node] - self.explored

    if len(to_explore) > 0: 
        self.explored = self.explored | to_explore
        
        for node2 in to_explore: 
            self.traverse(node2)
      
  def return_nodes(self): 
    return self.explored


netx = pd.read_csv('./../../data/genemania/GeneMania_AttentionWalk_input_all.csv')

graph = dict()

for A,B in netx.values: 
  if A not in graph.keys(): 
    graph[A] = set([B])
  else: 
    graph[A].add(B)
  
  # make it symetric  
  if B not in graph.keys(): 
    graph[B] = set([A])
  else: 
    graph[B].add(A)
  
has_explored = set()
sep_graphs = dict()

i = 0
fail = []
for node in graph.keys(): 
  print (f'Percentage of nodes traversed: {100*len(has_explored) / len(graph.keys()):.2f}', end='\r')
  if node not in has_explored: 
    try: 
        exp = explorer(graph, node)
        graph_nodes = exp.return_nodes()
        sep_graphs[i] = graph_nodes
        has_explored = has_explored|graph_nodes
        i+=1
    except:
        print() 
        fail.append(node)
        raise
    
print('failures: ')
[print(f'{x} : {graph[x]}') for x in fail]

num_graphs = len(sep_graphs.keys())
print('number of separate graphs: %d' %num_graphs)

for g in sep_graphs: 
  print('%d: length = %d' %(g, len(sep_graphs[g])))
