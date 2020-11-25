#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Assignment 1, Problem 2: Ring Detection

Team Number: 17 
Student Names: Alexander Lahti, Johan Haag
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
from typing import Set, Tuple  # noqa
import unittest  # noqa
from src.graph import Graph  # noqa
from src.ring_data import data  # noqa
# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger("put name here")
#   a = 5
#   logger.debug(f"a = {a}")
import logging  # noqa

__all__ = ['ring', 'ring_extended']


def ring(G: Graph) -> bool:
    """
    Sig:  Graph G(V, E) -> bool
    Pre: G is an undirected graph, with no duplicate edges
    Post: none
    Ex:   Sanity tests below
          ring(g1) = False
          ring(g2) = True
    """
    if G.nodes:
        return ring_help(G, [], G.nodes[0], G.nodes)
    else:
        return False


def ring_help(G: Graph, walk: [(str, str)], current_node: str, untouched_nodes: [str]) -> bool:
    """
    Sig:  Graph G(V,E), [(str, str)], str, [str] -> bool
    Pre: G is an undirected graph, with no duplicate edges, current_node has to be a node in G
    Post: walk will get modified with added edges if the function returns True.
    We will delete every node that we go through and don't find a ring to from untouched_nodes
    Ex:   g1 and g2 are from Sanity tests below
          ring_extended_help(g1, [], h, g1.nodes) = False
          ring_extended_help(g2, [], a, g1.nodes) = True
    """
    neighbors = G.neighbors(current_node)
    for n in neighbors:
        # Variant: len(neighbors) - (neighbors.index(n) - 1)
        if ((n, current_node) in walk) or ((current_node, n) in walk):
            continue
        for edge in walk:
            # Variant: len(walk) - (walk.index(edge) - 1)
            if n in edge:
                return True
        walk.append((n, current_node))
        if ring_help(G, walk, n, untouched_nodes):
        # Variant: Depth first search, if we encounter the same node twice we've found a ring and return True, 
        # if we run out of edges we go back (and remove current_node from untouched_nodes) until we find a new edge,
        # if we go all the way to the start we check if untouched_nodes is empty and if not, start over with the first node there.
            return True
        else:
            walk.pop()
    untouched_nodes.remove(current_node)
    if len(walk) == 0 and len(untouched_nodes) > 0:
        return ring_help(G, walk, untouched_nodes[0], untouched_nodes)
        # Variant: Depth first search, if we encounter the same node twice we've found a ring and return True, 
        # if we run out of edges we go back (and remove current_node from untouched_nodes) until we find a new edge,
        # if we go all the way to the start we check if untouched_nodes is empty and if not, start over with the first node there.
    else:
        return False
        

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


class RingTest(unittest.TestCase):
    """
    Test Suite for ring detection problem

    Any method named "test_something" will be run when this file is executed.
    Use the sanity check as a template for adding your own test cases if you
    wish. (You may delete this class from your submitted solution.)
    """
    logger = logging.getLogger('RingTest')

    def assertIsRing(self, graph, edges):
        """
        Asserts that a trail of edges is a ring in the graph
        """
        for e in edges:
            self.assertIn(
                e,
                graph,
                f"The edge {e} of the ring does not exist in the graph."
            )

        self.assertGreaterEqual(
            len(edges),
            3,
            "A ring consists of at least 3 edges."
        )

        for i, (u_i, v_i) in enumerate(edges[:-1]):
            u_j, v_j = edges[i+1]
            self.assertTrue(
                u_i in set([u_j, v_j]) or v_i in set([u_j, v_j]),
                f"The edges ('{u_i}', '{v_i}') and "
                f"('{u_j}', '{v_i}') are not connected."
            )

        u_1, v_1 = edges[0]
        u_k, v_k = edges[-1]

        self.assertTrue(
            u_k in set([u_1, v_1]) or v_k in set([u_1, v_1]),
            "The ring is not closed "
            f"[({u_1}, {v_1}), ..., ({u_k}, {v_k})]."
        )

        for i, (u_i, v_i) in enumerate(edges[:-1]):
            for u_j, v_j in edges[i+1:]:
                self.assertTrue(
                    u_i not in set([u_j, v_j]) or v_i not in set([u_j, v_j]),
                    f"The edges ({u_i}, {v_i}) and "
                    f"({u_j}, {v_i}) are not distinct."
                )

    def test_sanity(self):
        """
        Sanity Test

        This is a simple sanity check for your function;
        passing is not a guarantee of correctness.
        """
        edges = [
            ('x', 'y'), ('y', 'z'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('c', 'e'), ('c', 'f'),
            ('d', 'g'), ('d', 'h'), ('h', 'i')
        ]
        g1 = Graph(is_directed=False)
        for u, v in edges:
            g1.add_edge(u, v)
        self.assertFalse(ring(g1))
        g1.add_edge('g', 'i')
        self.assertTrue(ring(g1))

    def test_extended_sanity(self):
        """
        sanity test for returned ring

        This is a simple sanity check for your function;
        passing is not a guarantee of correctness.
        """
        edges = [
            ('x', 'y'), ('y', 'z'), ('a', 'b'), ('a', 'c'), ('a', 'f'), ('c', 'e'), ('c', 'f'),
            ('d', 'f'), ('d', 'g'), ('g', 'h'), ('f', 'h')
        ]
        g2 = Graph(is_directed=False)
        for u, v in edges:
            g2.add_edge(u, v)

        found, the_ring = ring_extended(g2)
        self.assertTrue(found)
        self.assertIsRing(g2, the_ring)

    def test_ring(self):
        """
        Test for ring

        passing is not a guarantee of correctness.
        """
        for i, instance in enumerate(data):
            graph = instance["graph"].copy()
            found = ring(graph)
            self.assertEqual(
                found,
                instance["expected"],
                f"instance[{i}] with {len(graph.nodes)} nodes"
            )

    def test_ring_extended(self):
        """
        Test for returned ring

        passing is not a guarantee of correctness.
        """
        for i, instance in enumerate(data):
            graph = instance["graph"].copy()
            found, the_ring = ring_extended(graph)
            self.assertEqual(
                found,
                instance["expected"],
                f"instance[{i}] with {len(graph.nodes)} nodes"
            )
            if instance["expected"]:
                self.assertIsRing(instance["graph"].copy(), the_ring)
            else:
                self.assertListEqual(the_ring, [])


if __name__ == '__main__':
    # Set logging config to show debug messages.
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
