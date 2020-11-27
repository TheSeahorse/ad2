#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Assignment 1, Problem 1: Weightlifting

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
from src.weightlifting_data import data  # noqa
from typing import List, Set  # noqa
import unittest  # noqa
# If you need to log information during tests, execution, or both,
# then you can use this library:
# Basic example:
#   logger = logging.getLogger("put name here")
#   a = 5
#   logger.debug(f"a = {a}")
import logging  # noqa

__all__ = ['weightlifting', 'weightlifting_subset']


def weightlifting(P: Set[int], weight: int) -> bool:
    '''
    Sig:  Set[int], int -> bool
    Pre:  P contains only non-negative integers, weight is a non-negative integer
    Post: (none)
    Ex:   P = {2, 32, 234, 35, 12332, 1, 7, 56}
          weightlifting(P, 299) = True
          weightlifting(P, 11) = False
    '''
    plate_list = list(P)
    if sum(plate_list) < weight:
        return False
    
    # Initialise the dynamic programming matrix, A
    matrix = [
        [None for i in range(weight + 1)] 
        for j in range(len(plate_list) + 1)
        ]
    counter = 0
    
    # VARIANT: len(matrix[0]) - (matrix[0].index(entry) - 1)
    for entry in matrix[0]:
        if counter == 0:
            matrix[0][counter] = True
        else:
            matrix[0][counter] = False
        counter += 1
    plate_list.sort()
    return weightlifting_help(plate_list, weight, 0, plate_list[0], matrix)


def weightlifting_help(P: list, weight: int, index: int, subset_sum: int, matrix: [list]) -> bool:
    '''
    Sig:  list, int, int, int, [list] -> bool
    Pre:  P contains only non-negative integers, weight is a non-negative integer, subset_sum is zero, index is zero
    Post: index takes the value of the amount of counted elements in P - 1,
          matrix is filled with boolean values that determines if P has a subset that sums to weight
    '''
    current_plate = P[index]
    current_weight = 0
    
    
    for entry in matrix[index + 1]:
        # VARIANT: len(matrix[index + 1]) - (matrix[index + 1].index(entry) - 1)
        lookup_weight = current_weight - current_plate
        if current_weight > subset_sum:
            matrix[index + 1][current_weight] = False
        elif matrix[index][current_weight]:
            matrix[index + 1][current_weight] = True
        elif lookup_weight < 0:
            matrix[index + 1][current_weight] = False
        else:
            matrix[index + 1][current_weight] = matrix[index][lookup_weight]
        current_weight += 1
    if matrix[index + 1][weight]:
        return True
    elif (len(P) - 1) == index:
        return False
    else:
        # VARIANT: (len(P) - 1) - index
        return weightlifting_help(P, weight, index + 1, subset_sum + P[index + 1],  matrix)
        

def weightlifting_subset(P: Set[int], weight: int) -> Set[int]:
    '''
    Sig:  Set[int], int -> Set[int]
    Pre:  P contains only non-negative integers, weight is a non-negative integer
    Post: (none)
    Ex:   P = {2, 32, 234, 35, 12332, 1, 7, 56}
          weightlifting_subset(P, 299) = {56, 7, 234, 2}
          weightlifting_subset(P, 11) = {}
    '''
    plate_list = list(P)
    if sum(plate_list) < weight:
        return set()
    
    # Initialise the dynamic programming matrix, A
    matrix = [
        [None for i in range(weight + 1)] 
        for j in range(len(plate_list) + 1)
        ]
    
    counter = 0
    
    # VARIANT: len(matrix[0]) - (matrix[0].index(entry) - 1)
    for entry in matrix[0]:
        if counter == 0:
            matrix[0][counter] = True
        else:
            matrix[0][counter] = False
        counter += 1
    plate_list.sort()
    subset_list = weightlifting_subset_help(plate_list, weight, 0, plate_list[0], matrix)
    return set(subset_list)


def weightlifting_subset_help(P: list, weight: int, index: int, subset_sum: int, matrix: [list]) -> list:
    '''
    Sig:  list, int, list, int, int, int -> list
    Pre:  P contains only non-negative integers, weight is a non-negative integer, subset_sum is zero, index is zero
    Post: index takes the value of the amount of counted elements in P - 1,
          matrix is filled with boolean values that determines if P has a subset that sums to weight
    '''
    current_plate = P[index]
    current_weight = 0
    
    # VARIANT: len(matrix[index + 1]) - (matrix[index + 1].index(entry) - 1)
    for entry in matrix[index + 1]:
        lookup_weight = current_weight - current_plate
        if current_weight > subset_sum:
            matrix[index + 1][current_weight] = False
        elif matrix[index][current_weight]:
            matrix[index + 1][current_weight] = True
        elif lookup_weight < 0:
            matrix[index + 1][current_weight] = False
        else:
            matrix[index + 1][current_weight] = matrix[index][lookup_weight]
        current_weight += 1
    if matrix[index + 1][weight]:
        return get_subset(P, weight, index + 1, matrix, [])
    elif (len(P) - 1) == index:
        return []
    else:
        # VARIANT: (len(P) - 1) - index
        return weightlifting_subset_help(P, weight, index + 1, (subset_sum + P[index + 1]), matrix)


def get_subset(P: list, current_weight: int, index: int, matrix: [list], subset: list) -> list:
    '''
    Sig:  list, int, int, [list], list -> list
    Pre:  P contains only non-negative integers, current_weight is a non-negative integer, index is the index of the last plate used
          matrix is filled with correct boolean values that determines the subset that sums to weight
    Post: current weight = 0, index is decreased by one for each recursive call, 'subset' contains the subset that sums to weight
    '''
    if current_weight == 0:
        return subset
    elif matrix[index - 1][current_weight]:
        # VARIANT: current_weight
        return get_subset(P, current_weight, index - 1, matrix, subset)
    else:
        subset.append(P[index - 1])
        # VARIANT: current_weight
        return get_subset(P, current_weight - P[index - 1], index - 1, matrix, subset)


class weightliftingTest(unittest.TestCase):
    """
    Test Suite for weightlifting problem

    Any method named "test_something" will be run when this file is executed.
    Use the sanity check as a template for adding your own test cases if you
    wish. (You may delete this class from your submitted solution.)
    """
    logger = logging.getLogger('WeightLiftingTest')

    def test_satisfy_sanity(self):
        """
        Sanity Test for weightlifting()

        passing is not a guarantee of correctness.
        """
        plates = {2, 32, 234, 35, 12332, 1, 7, 56}
        self.assertTrue(
            weightlifting(plates, 9)
        )
        self.assertFalse(
            weightlifting(plates, 11)
        )
        plates_2 = []
        for i in range(26):
            plates_2.append(i)
        self.assertFalse(
            weightlifting(set(plates_2), 111333)
        )

    def test_subset_sanity(self):
        """
        Sanity Test for weightlifting_subset()

        passing is not a guarantee of correctness.
        """
        plates = {2, 32, 234, 35, 12332, 1, 7, 56}
        weight = 8
        sub = weightlifting_subset(plates, weight)
        for p in sub:
            self.assertIn(p, plates)
        self.assertEqual(sum(sub), weight)

        weight = 11
        sub = weightlifting_subset(plates, weight)
        self.assertSetEqual(sub, set())

    def test_satisfy(self):
        for instance in data:
            self.assertEqual(
                weightlifting(instance["plates"], instance["weight"]),
                instance["expected"]
            )

    def test_subset(self):
        """
        Sanity Test for weightlifting_subset()

        passing is not a guarantee of correctness.
        """
        for instance in data:
            plates = weightlifting_subset(
                instance["plates"].copy(),
                instance["weight"]
            )
            self.assertEqual(type(plates), set)

            for plate in plates:
                self.assertIn(plate, instance["plates"])

            if instance["expected"]:
                self.assertEqual(
                    sum(plates),
                    instance["weight"]
                )
            else:
                self.assertSetEqual(
                    plates,
                    set()
                )


if __name__ == '__main__':
    # Set logging config to show debug messages.
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
