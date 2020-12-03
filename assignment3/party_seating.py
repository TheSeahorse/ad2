#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Assignment 3, Problem 2: Party Seating

Team Number: 26
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
from src.party_seating_data import data  # noqa
from typing import List, Tuple  # noqa
from src.graph import Graph # noqa
import unittest  # noqa

# If your solution needs a queue, then you can use this one:
from collections import deque  # noqa

# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger('put name here')
#   a = 5
#   logger.debug(f"a = {a}")
import logging  # noqa

__all__ = ['party']


def party(known: List[List[int]]) -> Tuple[bool, List[int], List[int]]:
    """
    Sig:  List[List[int]] -> Tuple[bool, List[int], List[int]]
    Pre:
    Post:
    Ex:   party([[1, 2], [0], [0]]) = True, [0], [1, 2]
    """
    if not known:
        return True, [], []
    G = Graph(is_directed = False)
    index = 0
    for person in known:
        for neighbor in person:
            if not G.__contains__((str(index), str(neighbor))):
                G.add_edge(str(index), str(neighbor))
        index += 1

    lonely_people = []
    for r in range(0, len(known)):
        if not str(r) in G.nodes:
            lonely_people.append(str(r))

    if not G.edges:
        return True, [], list_str_to_int(lonely_people)

    table1 = [G.edges[0][0]]
    table2 = lonely_people
    while(G.edges):
        if not find_tables(G, G.edges[0][0], table1, table2):
            return False, [], []
    
    
    return True, list_str_to_int(table1), list_str_to_int(table2)


def list_str_to_int(l: List[str]) -> List[int]:
    """
    Sig:  
    Pre:
    Post:
    Ex:   
    """
    int_list = []
    for i in l:
        int_list.append(int(i))
    return int_list


def find_tables(G: Graph, node: str, table1: List[str], table2: List[str]) -> bool:
    """
    Sig:  List[List[int]] -> Tuple[bool, List[int], List[int]]
    Pre: node exists in table1
    Post:
    Ex:   party([[1, 2], [0], [0]]) = True, [0], [1, 2]
    """
    if not G.neighbors(node):
        return True
    for n in G.neighbors(node):
        if n in table2:
            return False
        G.remove_edge(n, node)
        table2.append(n)
        if not find_tables(G, n, table2, table1):
            return False
    return True


class PartySeatingTest(unittest.TestCase):
    """
    Test suite for party seating problem
    """
    logger = logging.getLogger('PartySeatingTest')

    def known_test(self, known, A, B):
        self.assertEqual(
            len(A) + len(B),
            len(known),
            "wrong number of guests: "
            f"{len(known)} guests, "
            f"tables hold {len(A)} and {len(B)}"
        )
        for g in range(len(known)):
            self.assertTrue(
                g in A or g in B,
                f"Guest {g} not seated anywhere"
            )
        for a1, a2 in ((a1, a2) for a2 in A for a1 in A):
            self.assertNotIn(
                a2,
                known[a1],
                f"Guests {a1} and {a2} seated together, and know each other"
            )
        for b1, b2 in ((b1, b2) for b2 in B for b1 in B):
            self.assertNotIn(
                b2,
                known[b1],
                f"Guests {b1} and {b2} seated together, and know each other"
            )

    def test_sanity(self):
        """
        Sanity test

        A minimal test case.
        """
        known = [[1, 2], [0], [0]]
        _, A, B = party(known)
        self.known_test(known, A, B)

    def test_party(self):
        counter = 0
        for instance in data:
            known = instance["known"]
            expected = instance["expected"]
            success, A, B = party(known)
            print("counter: " + str(counter))
            print("known: " + str(known))
            print("expected: " + str(expected))
            print("our result: " + str(success) + "\ntable1: " + str(A), "\ntable2: " + str(B))

            if not expected:
                self.assertFalse(success)
                continue
            self.known_test(known, A, B)
            counter += 1


if __name__ == '__main__':
    # Set logging config to show debug messages.
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
