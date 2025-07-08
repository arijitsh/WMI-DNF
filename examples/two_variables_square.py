#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SimpleWMISolver import SimpleWMISolver
from utils.realsUniverse import RealsUniverse
from utils.weightFunction import WeightFunction
import numpy as np
import time


def main():
    np.random.seed(42)

    # Problem setup
    cntReals = 2  # Two real variables
    cntBools = 0  # No boolean variables

    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=30)


    expr = [
        [
            [(0,1), (1, 0), ('>=', 10)],
            [(0, 1), (1, 0), ('<=', 30)],
            [(0, 0), (1, 1), ('>=', 10)],
            [(0, 0), (1, 1), ('<=', 30)],
        ]
    ]
    monomials = [[1, [0, 0]]]

    poly_wf = WeightFunction(monomials, np.array([]))

    # Algorithm parameters
    eps = 0.2
    delta = 0.1


    # Check if temp directory exists, create if not
    if not os.path.exists("temp"):
        os.makedirs("temp")
        print("Created temp/ directory for LattE intermediate files.")

    timestamp_start = time.time()

    try:
        # Run WMI solver
        task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
        result = task.simpleCoverage(eps, delta)

        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        print(f"Result: {result:.6f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print()
        print("SUCCESS: LattE integration completed!")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
