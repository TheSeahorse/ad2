#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Assignment 2, Problem 2: Recomputing a Minimum Spanning Tree

Team Number:
Student Names:
'''

'''
Copyright: justin.pearson@it.uu.se and his teaching assistants, 2020.

This file is part of course 1DL231 at Uppsala University, Sweden.

Permission is hereby granted only to the registered students of that
course to use this file, for a homework assignment.

The copyright notice and permission notice above shall be included in
all copies and extensions of this file, and those are not allowed to
appear publicly on the internet, both during a course instance and
forever after.
'''
from src.recompute_mst_data import data  # noqa
from typing import Set, Tuple  # noqa
from src.graph import Graph  # noqa
import unittest  # noqa


# If your solution needs a queue (like the BFS algorithm),
# then you can use this one:
from collections import deque  # noqa

# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger("put name here")
#   a = 5
#   logger.debug(f"a = {a}")
import logging  # noqa

__all__ = ['update_MST_1', 'update_MST_2', 'update_MST_3', 'update_MST_4']


def update_MST_1(G: Graph, T: Graph, e: Tuple[str, str], weight: int):
    """
    Sig:  Graph G(V, E), Graph T(V, E), edge e, int ->
    Pre:
    Post:
    Ex:   TestCase 1 below
    """
    (u, v) = e
    assert(e in G and e not in T and weight > G.weight(u, v))


def update_MST_2(G: Graph, T: Graph, e: Tuple[str, str], weight: int):
    """
    Sig:  Graph G(V, E), Graph T(V, E), edge e, int ->
    Pre:
    Post:
    Ex:   TestCase 2 below
    """
    (u, v) = e
    assert(e in G and e not in T and weight < G.weight(u, v))
    T.add_edge(u, v)
    _, ring = ring_extended(T)

    max_edge = ('err', 'err')
    max_weight = 0
    for (u, v) in ring:
        if T.weight(u, v) and (T.weight(u, v) > max_weight):
            max_weight = T.weight(u, v)
            max_edge = (u, v)
    
    T.remove_edge(max_edge[0], max_edge[1])


def update_MST_3(G: Graph, T: Graph, e: Tuple[str, str], weight: int):
    """
    Sig:  Graph G(V, E), Graph T(V, E), edge e, int ->
    Pre:
    Post:
    Ex:   TestCase 3 below
    """
    (u, v) = e
    assert(e in G and e in T and weight < G.weight(u, v))


def update_MST_4(G: Graph, T: Graph, e: Tuple[str, str], weight: int):
    """
    Sig:  Graph G(V, E), Graph T(V, E), edge e, int ->
    Pre:
    Post:
    Ex:   TestCase 4 below
    """
    (u, v) = e
    assert(e in G and e in T and weight > G.weight(u, v))


def ring_extended(G: Graph) -> Tuple[bool, Set[Tuple[str, str]]]:
    """
    Sig:  Graph G(V,E) -> Tuple[bool, [(str, str)]]
    Pre: G is an undirected graph, with no duplicate edges
    Post: none
    Ex:   Sanity tests below
          ring_extended(g1) = False, []
          ring_extended(g2) = True, [('a','c'),('c','f'),
                                     ('f','h'),('h','g'),('g','d'),('d','f'),
                                     ('f','a')]
    """
    if G.nodes:
        return ring_extended_help(G, [], G.nodes[0], G.nodes)
    else:
        return (False, [])


def ring_extended_help(G: Graph, walk: [(str, str)], current_node: str, untouched_nodes: [str]) -> [bool, [(str, str)]]:
    """
    Sig:  Graph G(V,E), [(str, str)], str, [str] -> Tuple[bool, [(str, str)]]
    Pre: G is an undirected graph, with no duplicate edges, current_node has to be a node in G
    Post: walk will get modified so it becomes a ring if the function returns True
    We will delete every node that we go through and don't find a ring to from untouched_nodes
    Ex:   g1 and g2 are from Sanity tests below
          ring_extended_help(g1, [], h, g1.nodes) = False, []
          ring_extended_help(g2, [], a, g2.nodes) = True, [('a','c'),('c','f'),
                                     ('f','h'),('h','g'),('g','d'),('d','f'),
                                     ('f','a')]
    """
    neighbors = G.neighbors(current_node)
    for n in neighbors:
        # Variant: len(neighbors) - (neighbors.index(n) - 1)
        if ((n, current_node) in walk) or ((current_node, n) in walk):
            continue
        counter = 0
        for edge in walk:
            # Variant: len(walk) - (walk.index(edge) - 1)
            if n in edge:
                for x in range(counter):
                    walk.pop(0)
                walk.append((n, current_node))
                return (True, walk)
            counter += 1
        walk.append((n, current_node))
        result = ring_extended_help(G, walk, n, untouched_nodes)
        # Variant: Depth first search, if we encounter the same node twice we remove "tails" and return the ring, 
        # if we run out of edges we go back (and remove current_node from untouched_nodes) until we find a new edge,
        # if we go all the way to the start we check if untouched_nodes is empty and if not, start over with the first node there.
        if result[0]:
            return result
        else:
            walk.pop()
    untouched_nodes.remove(current_node)
    if len(walk) == 0 and len(untouched_nodes) > 0:
        return ring_extended_help(G, walk, untouched_nodes[0], untouched_nodes)
        # Variant: Depth first search, if we encounter the same node twice we remove "tails" and return the ring, 
        # if we run out of edges we go back (and remove current_node from untouched_nodes) until we find a new edge,
        # if we go all the way to the start we check if untouched_nodes is empty and if not, start over with the first node there.
    else:
        return (False, [])



class RecomputeMstTest(unittest.TestCase):
    """
    Test Suite for minimum spanning tree problem

    Any method named "test_something" will be run when this file is
    executed. You may add your own test cases if you wish.
    (You may delete this class from your submitted solution.)
    """
    logger = logging.getLogger('RecomputeMstTest')

    def assertUndirectedEdgesEqual(self, actual, expected):
        self.assertListEqual(
            sorted(((min(u, v), max(u, v)) for u, v in actual)),
            sorted(((min(u, v), max(u, v)) for u, v in expected))
        )

    def assertGraphIsConnected(self, graph):
        if len(graph.nodes) == 0:
            return
        visited = set()
        s = graph.nodes[0]
        queue = deque([s])
        while len(queue) > 0:
            u = queue.popleft()
            visited.add(u)
            for v in graph.neighbors(u):
                if v not in visited:
                    queue.append(v)
        for u in graph.nodes:
            self.assertIn(u, visited)

    def assertEdgesInGraph(self, edges, graph):
        for edge in edges:
            self.assertIn(edge, graph)

    def test_mst1(self):
        # TestCase 1: e in graph.edges and e not in tree.edges and
        #             weight > graph.weight(u, v)
        i = 0
        for instance in data:
            graph = instance['graph'].copy()
            tree = instance['mst'].copy()
            u, v = instance['solutions'][i]['edge']
            weight = instance['solutions'][i]['weight']
            expected = instance['solutions'][i]['expected']
            update_MST_1(graph, tree, (u, v), weight)
            self.assertUndirectedEdgesEqual(
                tree.edges,
                expected
            )

    def test_mst2(self):
        # TestCase 2: e in graph.edges and not e in tree.edges and
        #             weight < graph.weight(u, v)
        i = 1
        counter = 0
        for instance in data:
            graph = instance['graph'].copy()
            tree = instance['mst'].copy()
            u, v = instance['solutions'][i]['edge']
            weight = instance['solutions'][i]['weight']
            expected = instance['solutions'][i]['expected']
            update_MST_2(graph, tree, (u, v), weight)
            print(counter)
            print("edge:     " + u + v)
            print("expected: " + str(sorted(expected)))
            print("actual:   " + str(sorted(tree.edges)))
            print("============")
            self.assertUndirectedEdgesEqual(
                tree.edges,
                expected
            )
            counter += 1

    def test_mst3(self):
        # TestCase 3: e in graph.edges and e in tree and
        #             weight < graph.weight(u, v)
        i = 2
        for instance in data:
            graph = instance['graph'].copy()
            tree = instance['mst'].copy()
            u, v = instance['solutions'][i]['edge']
            weight = instance['solutions'][i]['weight']
            expected = instance['solutions'][i]['expected']
            update_MST_3(graph, tree, (u, v), weight)
            self.assertUndirectedEdgesEqual(
                tree.edges,
                expected
            )

    def test_mst4(self):
        # TestCase 4: e in graph.edges and e in tree and
        #             weight > graph.weight(u, v)
        i = 3
        for instance in data:
            graph = instance['graph'].copy()
            tree = instance['mst'].copy()
            u, v = instance['solutions'][i]['edge']
            weight = instance['solutions'][i]['weight']
            expected = instance['solutions'][i]['expected']
            expected_G = graph.copy()
            expected_G.set_weight(u, v, weight)
            update_MST_4(graph, tree, (u, v), weight)
            self.assertEdgesInGraph(tree.edges, expected_G)
            self.assertGraphIsConnected(tree)
            self.assertEqual(
                sum(graph.weight(u, v) for u, v in tree.edges),
                sum(graph.weight(u, v) for u, v in expected)
            )
            for u, v in tree.edges:
                self.assertEqual(
                    tree.weight(u, v),
                    expected_G.weight(u, v)
                )


if __name__ == '__main__':
    # Set logging config to show debug messages.
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
