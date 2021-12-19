from collections import defaultdict
from datetime import datetime
import json
from tqdm import tqdm
from sys import getsizeof
import heapq
from tqdm import tqdm
from collections import defaultdict
import heapq as heap
import numpy as np
import itertools
import math
from tabulate import tabulate


class Graph:
    '''
    this is the class that represent the graph object it is builded starting from a dictionary. The graph is represented
    by a dictionary of dict. {node_1: {node_2: {dictionary of attributes}}}. Where node_1 and node_2 are adjacent nodes
    and the dictionary of attributes represents the attributes of the edges such as the weight or the date etc.
    '''

    def __init__(self, graph={}):
        '''
        :param graph: a dictionary that represent the graph. Default is and empty dictionary

        this is the constructor of our class
        '''
        self.graph = graph

    def add_vertex(self, v):
        '''

        :param v: a node that has to be added to the graph

        this method add a node to the graph if and only if the node is not yet in the graph
        '''
        if v not in self.graph:
            self.graph[v] = dict()
        # else:
        #    #print("Node already in the graph")

    def add_edge(self, v1, v2):
        '''

        :param v1: first node
        :param v2: second node

        this method create an edge that goes from v1 to v2
        '''
        # if len(self.graph[v1]) == 0:
        if [v2, 0] not in self.graph[v1]:
            self.graph[v1].append([v2, 0])
        else:
            for edge in self.graph[v1]:
                if v2 in edge:
                    edge[1] += 1

    def add_edge_with_attr(self, v1, v2, **attr):
        '''

        :param v1: first node
        :param v2: second node
        :param attr: attributes that you want to add to the egde. Example weight = 1, date = 20211219

        this method create an edge that goes from v1 to v2, whit all the attributes that are given in input.
        If the second node is not yet in the adjacency list of the first one an edge will be created and the weight will
        be set to 1, otherwise the weight will be incremented by one. If two node have already interacted previously the
        date of the new interaction will be added to the attributes date.
        '''
        # controllo se v2 è tra gli adiacenti di v1
        if v2 not in self.graph[v1]:
            # se non è tra gli adiacenti
            if 'weight' in attr:
                # lo aggiungo con tutti gli attributi
                self.graph[v1][v2] = attr
            else:
                # se weight non è tra gli attr aggiungo anche l'attributo weight a 0
                self.graph[v1][v2] = attr
                self.graph[v1][v2]['weight'] = 1
        else:
            for k, v in attr.items():
                # se gli sto passando una nuova data allo stesso arco appendi la data alla lista
                # di date
                if k == 'date':
                    if (len(self.graph[v1][v2][k]) > 0) and (v not in self.graph[v1][v2][k]):
                        self.graph[v1][v2][k] += ',' + v
                    else:
                        self.graph[v1][v2][k] = v
                else:
                    self.graph[v1][v2][k] = v
            self.graph[v1][v2]['weight'] += 1
            # return graph

    def get_graph(self):
        '''

        :return: return the dictionary that represents the graph
        '''
        return self.graph

    def get_edges(self, vertex=''):
        '''

        :param vertex: the node of which we want the arcs . If it is empty return all the edges of the graph
        :return: return a list of tuples that represents the edges
        '''
        list_edges = []
        if vertex == '':
            for vrtx in self.graph.keys():
                for adj in self.graph[vrtx].keys():
                    list_edges.append((vrtx, adj))
        else:
            for adj in self.graph[vertex].keys():
                list_edges.append((vertex, adj))

        return list_edges

    def get_adj(self, vertex):
        '''

        :param vertex: the node of which we want the arcs
        :return: return the list of the adjacents
        '''
        return list(self.graph[vertex].keys())

    def get_edges_old(self, vertex=''):
        list_edges = []
        if vertex == '':
            for vertex, edges in self.graph.items():
                for edge in edges:
                    list_edges.append((vertex, edge[0]))
        else:
            for edge in self.graph[vertex]:
                list_edges.append((vertex, edge[0]))
        return list_edges

    def get_vertex(self):
        '''

        :return: return the list of nodes of the graph
        '''
        return list(self.graph.keys())

    def get_edges_with_labels(self, v1='', v2=''):
        '''

        :param v1: first node. Default is empty
        :param v2: second node. Default is empty
        :return: retutn a dictionary that has as key the couple (v1,v2) and as value the dictionary of attributes of the
                edge between v1 and v2
        '''
        dict_edges = {}
        if v1 == '' and v2 == '':
            for vrtx in self.get_vertex():
                for adj in self.get_adj(vrtx):
                    edge_with_labels = self.get_graph()[vrtx][adj]
                    dict_edges[vrtx, adj] = (edge_with_labels)
        else:
            # se gli passo i nodi voglio il dizionario di labels per quell'arco
            edge_with_labels = self.get_graph()[v1][v2]
            dict_edges = edge_with_labels
        return dict_edges

    def check_directed(self):
        '''

        :return: check if the graph is directed or not. Iterate over the nodes of the graph when we find a couple of node
        such that node1 has node2 in the adjacency list but node2 is not in the adjacency list of node1 then we say that the
        graph is directed
        '''
        i = 0
        while i < len(self.get_vertex()):
            undirected = True
            A = self.get_vertex()[i]
            adj_a = self.get_adj(A)
            for B in adj_a:
                for c in self.get_adj(B):
                    if A != c:
                        undirected = False
                        break
            if not undirected:
                break
            i += 1
        return not undirected

    def func_1(self):
        '''

        :return: method that compute some metrics about the graph. It compute if the graph is directed, the number of nodes,
        the number of edges, the density degree of the graph (#edges/#all the possible edges) and if the graph is sparse or
        not that depends on the desity degree of the graph. if the density is between 0 and 0.5 then the graph is sparse,
        otherwise is dense.
        '''
        directed = self.check_directed()
        number_users = len(self.get_vertex())
        number_answers = len(self.get_edges())
        s = 0
        for i in self.get_vertex():
            n_edges = len(self.get_adj(i))
            s += n_edges
        average_links = s / len(self.get_vertex())
        density_degree = number_answers / (number_users * (number_users - 1))
        sparse = 0 < density_degree <= 0.5
        table = [['Directed', '# of users', '# of answers/comments', 'Average # of links', 'Density degree', 'Sparse?'], [directed, number_users, number_answers, average_links, density_degree, sparse]]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        #print('Directed: ', directed)
        #print('Number of users: ', number_users)
        #print('Number of answers/comments: ', number_answers)
        #print('Average number of links per user: ', average_links)
        #print('Density degree of the graph: ', density_degree)
        #print('Sparse: ', sparse)

    def func_2(self, node, start, end, centrality):
        '''

        :param node: a node of which want to know the centrality
        :param start: the begin of an interval of time
        :param end: the end of an interval of time
        :param centrality: the type of centrality
        :return: return the centrality measure
        '''
        self.get_graph_in_intervall(start, end)
        if centrality == 'DegreeCentrality':
            centr = self.get_degree_centrality(node)
        elif centrality == 'ClosenessCentrality':
            centr = self.closeness_centrality(node)
        elif centrality == 'Betweeness':
            centr = self.betweenness(node)
        elif centrality == 'PageRank':
            print('insert number of steps: ')
            n_iterazioni = int(input())
            fill_graph(self)
            g_supp = create_full_graph(self)
            P = create_prob_matrix(self, g_supp, 0.1)
            page_rank_vector = page_rank(P, node, 1090, self.get_vertex())
            index = self.get_vertex().index(node)
            centr = page_rank_vector[index]
            
        table = [['Interval', 'User', 'Metric', 'Metric score'], [start + '-' + end, node, centrality, centr]]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        return centr
    def get_graph_in_intervall(self, start, end):
        '''

        :param start: the begin of an interval of time
        :param end: the end of an interval of time
        :return: return a subgraph with the edges that have the label date between start and end
        '''
        for k, v in self.get_edges_with_labels().items():
            to_remove = []
            list_date = v['date'].split(',')
            if len(list_date) > 0:
                for i in v['date'].split(','):
                    if i > end or i < start:
                        to_remove.append(i)
                if len(to_remove) == len(list_date):
                    self.graph[k[0]].pop(k[1])

                else:
                    for date_to_remove in to_remove:
                        self.graph[k[0]][k[1]]['date'] = self.graph[k[0]][k[1]]['date'].replace(date_to_remove, '')
                        if self.graph[k[0]][k[1]]['date'].startswith(','):
                            self.graph[k[0]][k[1]]['date'] = self.graph[k[0]][k[1]]['date'].lstrip(',')
                        self.graph[k[0]][k[1]]['weight'] -= 1


    #def get_weight(self, vertex, edge):
    #    edges = self.graph[vertex]
    #    print(edges)
    #    for e in edges:
    #        if e[0] == edge:
    #            return e[1]

    def get_degree_centrality(self, node):
        '''

        :param node: a node of the graph
        :return: the degree centrality of the node. That is computed as (in degree + out degree)/ number of nodes in the graph
        '''
        if len(self.graph[node]) == 0:
            return 'User not in the graph'
        in_degree = 0
        out_degree = 0
        for i in self.get_edges():
            if node == i[1]:
                in_degree += 1
            if node == i[0]:
                out_degree += 1
        return (in_degree + out_degree) / (len(self.get_vertex()) - 1)

    def closeness_centrality(self, node):
        '''

        :param node: a node of the graph
        :return: the closeness centrality of the node. That is computed as number of nodes/he shortest-path distance that starts from node
        '''
        if node not in self.graph.keys() == 0:
            return 'User not in the graph'
        parents_tree, nodeCosts = dijkstra(self, node)
        sum_distances = 0
        for k, v in nodeCosts.items():
            sum_distances += v
        if sum_distances > 0:
            out = (len(self.get_vertex())-1)/sum_distances
            return out/(len(self.get_vertex())-1)
        else:
            return 0

    def betweenness(self, node):

        """
        :param G: a graph
        :param node: a node whose betweenness centrality we want to compute
        :return:
            score / normalization: normalized betweenness value
        """
        
        # get list of nodes
        nodes = self.get_vertex()

        # compute all pairs
        pairs = list(itertools.permutations(nodes, 2))

        score = 0

        # find shortest path for each pair of nodes
        for i in pairs:

            # number of paths
            total_paths = 0
            paths_on_node = 0

            start = i[0]
            end = i[1]

            d = dijkstra_custom(self, start, end)[0]

            # shortest_paths = i, "-->", d
            # print(shortest_paths)

            # if a path exists, add it to the count of total paths
            if d != 'T':
                total_paths += 1

                # count paths that pass through the node (removing endpoints)
                for vertex in d[1:-1]:
                    if vertex == node:
                        paths_on_node += 1

            if total_paths != 0:
                score += (paths_on_node / total_paths)

        # compute betweenness as sum
        # betweenness = paths_on_node / total_paths

        normalization = (len(nodes) - 1) * (len(nodes) - 2)

        # betweenness_normalized = betweenness / normalization

        return score / normalization


def dijkstra(G: Graph, startingNode):
    '''

    :param G: a graph
    :param startingNode: the starting node
    :return:
        parents_tree: a dictionary that represent the path from the starting node to the end of the path
        n_costs: a dictionary that has as key a node in the path and as value the cost to reach the node
    '''
    visited = set()
    parents_tree = defaultdict(list)
    to_visit = []
    n_costs = defaultdict(lambda: float('inf'))
    # if startingNode in G.get_adj(startingNode):
    #    nodeCosts[startingNode]=g_test.get_edges_with_labels(startingNode,startingNode)['weight']
    # else:
    n_costs[startingNode] = 0
    heap.heappush(to_visit, startingNode)

    while to_visit:
        node = heap.heappop(to_visit)
        visited.add(node)

        for adj_node, adj_labels in G.get_graph()[node].items():
            weight = adj_labels['weight']
            if adj_node in visited:
                # if adjNode == node:
                #    print(adjNode)
                #    #se il nodo ha un arco che rientra in se stesso
                #    nodeCosts[adjNode] = g_test.get_edges_with_labels(adjNode,adjNode)['weight']
                continue
            new_cost = n_costs[node] + weight
            if n_costs[adj_node] > new_cost:
                parents_tree[node].append(adj_node)
                n_costs[adj_node] = new_cost
                heap.heappush(to_visit, adj_node)

    return parents_tree, n_costs


def page_rank(P, node, range_, vertex_list):
    '''

    :param P: the probability matrix of the graph
    :param node: The node for which we'd like to find its PageRank value
    :param range_: number of steps after we want to know the pagerank
    :param vertex_list: the list of nodes of the graph
    :return:
        if the probabilities converge return the vector. at time t
        If not return all the vector for every time
    '''
    q_i_prev = np.array([1 if i == node else 0 for i in vertex_list])
    q_seq = []
    conv = False
    for j in range(range_):
        q_i = []
        for i in P.T:
            q_i.append(np.dot(q_i_prev, i))
        q_seq.append(q_i)
        if (q_i_prev == q_i).all():
            print('converge')
            conv = True
            return q_i
        q_i_prev = np.array(q_i)
    if not conv:
        print('not converge in {} steps'.format(range_))
        return q_seq


def fill_graph(g: Graph):
    '''

    :param g: a graph

    this function add to sinks one edge that reach every other node in the graph
    '''
    s = []
    n_s = []
    for k, v in g.get_graph().items():
        if len(v) == 0:
            s.append(k)
        else:
            n_s.append(k)
    val = dict.fromkeys(g.get_vertex(), {'weight': 1})
    for i in s:
        g.graph[i] = val
    print('#sink', len(s))


def add_labels(g: Graph, node1, node2, **attrs):
    '''

    :param g: graph
    :param node1: first node of an edge
    :param node2: second node of an edge
    :param attrs: the list of attributes that have to be added to the egde
    :return: the graph with the new attributes
    '''
    for k, v in attrs.items():
        g.get_graph()[node1][node2][k] = v
    return g


def graph_to_markov(g_test: Graph, alpha=0):
    if alpha and alpha <= 1:
        for i in g_test.get_edges_with_labels().keys():
            p = alpha * (1 / (len(g_test.get_adj(i[0]))))
            add_labels(g_test, i[0], i[1], prob=p)
    else:
        for i in g_test.get_edges_with_labels().keys():
            p = 1 / (len(g_test.get_adj(i[0])))
            add_labels(g_test, i[0], i[1], prob=p)


def dict_to_matrix(g: Graph):
    '''

    :param g: a graph
    :return: the graph represented as matrix
    '''
    l_m = []
    for i in range(len(g.get_vertex())):
        l_riga = []
        for j in range(len(g.get_vertex())):
            v1 = g.get_vertex()[i]
            v2 = g.get_vertex()[j]
            if g.get_vertex()[j] in g.get_adj(g.get_vertex()[i]):
                l_riga.append(1)
            else:
                l_riga.append(0)
        l_m.append(l_riga)
    l_m = np.array(l_m)
    new_a = []
    for i in l_m:
        n_edges = len(np.where(i == 1)[0])
        if n_edges > 0:
            a = i * (1 / n_edges)
        else:
            a = i
        new_a.append(a)
    new_a = np.array(new_a)
    return new_a


def create_full_graph(g: Graph):
    '''

    :param g: a graph
    :return: a graph that has the same nodes of the input one, but has edges from every node to every other node
    '''
    g_supp = Graph()
    g_supp.get_graph()
    for n1 in g.get_vertex():
        g_supp.add_vertex(n1)
    for n1 in g.get_vertex():
        for n2 in g.get_vertex():
            g_supp.add_edge_with_attr(n1, n2)
    return g_supp


def create_prob_matrix(g: Graph, g_supp: Graph, alpha):
    '''

    :param g: a graph
    :param g_supp: a graph that has the same nodes of the input one, but has edges from every node to every other node
    :param alpha: the probability of teleportation
    :return: the probability matrix for the graph g. In every position ij of the matrix there is the probability to go from
            i to j
    '''
    A = dict_to_matrix(g)
    M_ONES = dict_to_matrix(g_supp)
    P = alpha * M_ONES + (1 - alpha) * A
    return P


def dijkstra_custom(G, startingNode, endingNode):
    '''
    :param G: a graph
    :param startingNode: the starting node
    :param endingNode: the ending node
    :return:
        path: a list with the nodes in the path from the starting node to the ending Node
        shortest_distance[endingNode]: a scalar representing the distance of that path
    '''
    visited = set()
    to_visit = G.get_vertex()
    shortest_distance = {}
    parents = {}
    max_dist = math.inf

    # Initialize the distance with other nodes to + Inf
    for vertex in to_visit:
        shortest_distance[vertex] = math.inf
        parents[vertex] = None
    # initialize the distance of the starting node to zero
    shortest_distance[startingNode] = 0

    # Push the startingNode onto the to_visit
    heap.heappush(to_visit, startingNode)

    # as long as to_visit is not empty
    while to_visit:

        # pop smallest itam from to_visit
        node = heap.heappop(to_visit)

        # label the node as visited
        visited.add(node)

        # for each neighbour
        for adj_node, adj_labels in G.get_graph()[node].items():
            # get the weight of the edge between the node and each neighbour
            weight = adj_labels['weight']

            # calculate a new distance (new cost) for each neighbors
            new_cost = shortest_distance[node] + weight

            # update, if the new distance is smaller than the previous one
            if shortest_distance[adj_node] > new_cost:
                parents[adj_node] = node
                shortest_distance[adj_node] = new_cost
                heap.heappush(to_visit, adj_node)

    # case in which no path exists
    if parents[endingNode] == None:
        return ('There is no path')

    # list with the path from the startingNode to the endingNode
    path = [endingNode]
    node_now = endingNode

    while node_now != startingNode:
        path.append(parents[node_now])
        node_now = parents[node_now]
    path.reverse()

    ## result (Distance and the path)
    return path, shortest_distance[endingNode]