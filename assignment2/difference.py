#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Assignment 2, Problem 1: Search String Replacement

Team Number: 34
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
from src.difference_data import data  # noqa
from typing import Dict, Tuple  # noqa
import math  # noqa
import unittest  # noqa
from collections import defaultdict  # noqa
from string import ascii_lowercase  # noqa

# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger("put name here")
#   a = 5
#   logger.debug(f"a = {a}")
import logging  # noqa


# Solution to Task B:
def min_difference(u: str, r: str, R: Dict[str, Dict[str, int]]) -> int:
    """
    Sig:  str, str, Dict[str, Dict[str, int]] -> int
    Pre:  For all characters c in u and k in r,
          then R[c][k] exists, and R[k][c] exists.
    Post: (none)
    Ex:   Let R be the resemblance matrix
          min_difference("dinamck", "dynamic", R) --> 3
    """
    # To get the resemblance between two letters, use code like this:
    # difference = R['a']['b']
    
    dp_matrix = get_dp_matrix(u, r, R)

    return dp_matrix[len(u)][len(r)]


# Solution to Task C:
def min_difference_align(u: str, r: str,
                         R: Dict[str, Dict[str, int]]) -> Tuple[int, str, str]:
    """
    Sig:  str, str, Dict[str, Dict[str, int]] -> Tuple[int, str, str]
    Pre:  For all characters c in u and k in r,
          then R[c][k] exists, and R[k][c] exists.
    Post: (none)
    Ex:   Let R be the resemblance matrix
          min_difference_align("dinamck", "dynamic", R) -->
                                    3, "dinam-ck", "dynamic-"
    """
    dp_matrix = get_dp_matrix(u, r, R)
    
    result = get_aligned_strings(u, r, dp_matrix, "", "", (len(u), len(r)))
    return (dp_matrix[len(u)][len(r)], result[0], result[1])


def get_aligned_strings(u:str, r:str, dp_matrix:[list], u_aligned:str, r_aligned:str, index:Tuple[int, int]) -> Tuple[str, str]:
    """
    Sig:  str, str, [list], str, str, Tuple[int, int] -> Tuple[str, str]
    Pre:  u_aligned and r_aligned is empty string
          dp_matrix is a complete dynamic programming matrix for u and r
          index is a tuple of length(u) and length(r)
    Post: u_aligned and r_aligned is the positioning of u and r
          index == [0, 0]
    Ex:   get_aligned_strings("kz", "zm", [[0, 3, 5], [3, 6, 5], [6, 7, 8]], "", "", (0,0)) -> ('-kf', 'zm-')
    """
    i = index[0]
    j = index[1]
    
    if i == 0 and j == 0:
        return (u_aligned, r_aligned)
    
    if i == 0:
        new_u_aligned = '-' + u_aligned
        new_r_aligned = r[j-1] + r_aligned
        return get_aligned_strings(u, r, dp_matrix, new_u_aligned, new_r_aligned, (i, j-1))
        # VARIANT: i, j

    if j == 0:
        new_u_aligned = u[i-1] + u_aligned
        new_r_aligned = '-' + r_aligned
        return get_aligned_strings(u, r, dp_matrix, new_u_aligned, new_r_aligned, (i-1, j))
        # VARIANT: i, j

    minimum_value = min(dp_matrix[i-1][j-1],
                        dp_matrix[i][j-1],
                        dp_matrix[i-1][j])
    
    if minimum_value == dp_matrix[i-1][j-1]:
        new_u_aligned = u[i-1] + u_aligned
        new_r_aligned = r[j-1] + r_aligned
        return get_aligned_strings(u, r, dp_matrix, new_u_aligned, new_r_aligned, (i-1, j-1))
        # VARIANT: i, j
    
    if minimum_value == dp_matrix[i][j-1]:
        new_u_aligned = '-' + u_aligned
        new_r_aligned = r[j-1] + r_aligned
        return get_aligned_strings(u, r, dp_matrix, new_u_aligned, new_r_aligned, (i, j-1))
        # VARIANT: i, j

    if minimum_value == dp_matrix[i-1][j]:
        new_u_aligned = u[i-1] + u_aligned
        new_r_aligned = '-' + r_aligned
        return get_aligned_strings(u, r, dp_matrix, new_u_aligned, new_r_aligned, (i-1, j))
        # VARIANT: i, j


def get_dp_matrix(u: str, r: str, R: Dict[str, Dict[str, int]]) -> [list]:
    """
    Sig:  str, str, Dict[str, Dict[str, int]] -> Tuple[int, str, str]
    Pre:  For all characters c in u and k in r,
          then R[c][k] exists, and R[k][c] exists.
    Post: (none)
    Ex:   Let R be the resemblance matrix
          get_dp_matrix("kz", "zm", R) --> [[0, 3, 5], [3, 6, 5], [6, 7, 8]])
    """
    dp_matrix = [
        [0 for i in range(len(r) + 1)]
        # VARIANT: (len(r) + 1) - i
        for j in range(len(u) + 1)
        # VARIANT: (len(u) + 1) - j
    ]
    
    for i in range(1, len(u) + 1):
    # VARIANT: (len(u) + 1) - i
        for j in range(1, len(r) + 1):
        # VARIANT: (len(r) + 1) - j
            # Initializing matrix with cost of empty r matching u and cost of empty u matching r
            dp_matrix[i][0] = dp_matrix[i - 1][0] + R[u[i - 1]]['-']
            dp_matrix[0][j] = dp_matrix[0][j - 1] + R['-'][r[j - 1]]
    

    for i in range(1, len(u) + 1):
    # VARIANT: (len(u) + 1) - i
        for j in range(1, len(r) + 1):
        # VARIANT: (len(r) + 1) - j
            if(u[i-1] == r[j-1]):
                # If the characters are the same, we will use them with cost 0
                dp_matrix[i][j] = dp_matrix[i-1][j-1]
            else:
                # Else get the minimum cost from either, the cost of strings u[0..i - 1] and r[0..j - 1] + substitution cost, 
                # or the cost of a skip in u for strings u[0..i] and r[0..j - 1] + skip in u cost, 
                # or the cost of a skip in r for strings u[0..i - 1] and r[0..j] + skip in r cost
                dp_matrix[i][j] = min(dp_matrix[i-1][j-1] + R[u[i - 1]][r[j - 1]], 
                                      dp_matrix[i][j-1] + R['-'][r[j - 1]], 
                                      dp_matrix[i-1][j] + R[u[i - 1]]['-'])

    return dp_matrix



# Sample matrix provided by us:
def qwerty_distance() -> Dict[str, Dict[str, int]]:
    """
    Generates a QWERTY Manhattan distance resemblance matrix

    Costs for letter pairs are based on the Manhattan distance of the
    corresponding keys on a standard QWERTY keyboard.
    Costs for skipping a character depends on its placement on the keyboard:
    adding a character has a higher cost for keys on the outer edges,
    deleting a character has a higher cost for keys near the middle.

    Usage:
        R = qwerty_distance()
        R['a']['b']  # result: 5
    """
    R = defaultdict(dict)
    R['-']['-'] = 0
    zones = ["dfghjk", "ertyuislcvbnm", "qwazxpo"]
    keyboard = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    for row, content in enumerate(zones):
        for char in content:
            R['-'][char] = row + 1
            R[char]['-'] = 3 - row
    for a, b in ((a, b) for b in ascii_lowercase for a in ascii_lowercase):
        row_a, pos_a = next(
            (row, content.index(a))
            for row, content in enumerate(keyboard) if a in content
        )
        row_b, pos_b = next(
            (row, content.index(b))
            for row, content in enumerate(keyboard) if b in content
        )
        R[a][b] = int(
            math.fabs(row_b - row_a) + math.fabs(pos_a - pos_b)
        )
    return R


class MinDifferenceTest(unittest.TestCase):
    """
    Test Suite for search string replacement problem

    Any method named "test_something" will be run when this file is
    executed. Use the sanity check as a template for adding your own test
    cases if you wish.
    (You may delete this class from your submitted solution.)
    """
    logger = logging.getLogger('MinDifferenceTest')

    def test_diff_sanity(self):
        """
        Difference sanity test

        Given a simple resemblance matrix, test that the reported
        difference is the expected minimum. Do NOT assume we will always
        use this resemblance matrix when testing!
        """
        alphabet = ascii_lowercase + '-'
        # The simplest (reasonable) resemblance matrix:
        R = {
            a: {b: (0 if a == b else 1) for b in alphabet} for a in alphabet
        }
        # Warning: we may (read: 'will') use another matrix!
        self.assertEqual(min_difference("dinamck", "dynamic", R), 3)

    def test_align_sanity(self):
        """
        Simple alignment

        Passes if the returned alignment matches the expected one.
        """
        # QWERTY resemblance matrix:
        R = qwerty_distance()
        difference, u, r = min_difference_align(
            "polynomial",
            "exponential",
            R
        )
        # Warning: we may (read: 'will') use another matrix!
        self.assertEqual(difference, 15)
        # Warning: there may be other optimal matchings!
        if u != '--polyn-om-ial':
            self.logger.warning(f"'{u}' != '--polyn-om-ial'")
        if r != 'exp-o-ne-ntial':
            self.logger.warning(f"'{r}' != 'exp-o-ne-ntial'")


    def test_min_difference(self):
        R = qwerty_distance()
        for instance in data:
            difference = min_difference(
                instance["u"],
                instance["r"],
                R
            )
            self.assertEqual(instance["expected"], difference)

    def test_min_difference_align(self):
        R = qwerty_distance()
        for instance in data:
            difference, u, r = min_difference_align(
                instance["u"],
                instance["r"],
                R
            )

            self.assertEqual(instance["expected"], difference)
            self.assertEqual(len(u), len(r))
            u_diff, _, _ = min_difference_align(u, instance["u"], R)
            self.assertEqual(u_diff, 0)
            r_diff, _, _ = min_difference_align(r, instance["r"], R)
            self.assertEqual(r_diff, 0)


if __name__ == '__main__':
    # Set logging config to show debug messages.
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
