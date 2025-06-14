#!/usr/bin/env python3
import sys
import argparse

def parse_grid_test(filename):
    """
    Parses a test file containing several test cases.
    Each test case is preceded by the line "Testcase" and consists of grid rows.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    test_cases = []
    current_case = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Testcases"):
            continue
        if line == "Testcase":
            if current_case:
                test_cases.append(current_case)
                current_case = []
        else:
            current_case.append(line)
    if current_case:
        test_cases.append(current_case)
    return test_cases

def build_free_cells_from_test(grid):
    """
    Builds the ordering of free cells from the test grid.
    In the test grid, any cell that is not a wall ('W') is considered free.
    """
    free_cells = {}
    counter = 0
    for i, row in enumerate(grid):
        tokens = row.split()
        for j, token in enumerate(tokens):
            if token == 'W':
                continue
            free_cells[(i, j)] = counter
            counter += 1
    return free_cells

def find_agent(grid):
    """
    Searches for the agent symbol among: '>', '<', '^', 'v'.
    Returns a tuple (row, col, orientation) where:
      0: up, 1: right, 2: down, 3: left.
    """
    agent_symbols = {'>', '<', '^', 'v'}
    for i, row in enumerate(grid):
        tokens = row.split()
        for j, token in enumerate(tokens):
            if token in agent_symbols:
                if token == '^':
                    orient = 0
                elif token == '>':
                    orient = 1
                elif token == 'v':
                    orient = 2
                elif token == '<':
                    orient = 3
                return (i, j, orient)
    # Fallback: if no agent symbol is found, search for 's'
    for i, row in enumerate(grid):
        tokens = row.split()
        for j, token in enumerate(tokens):
            if token == 's':
                return (i, j, 0)  # default orientation if unspecified
    return None

def has_key_in_grid(grid):
    """
    Determines if the key ('k') is present in the test grid.
    If so, the agent has not yet picked it up (flag 0);
    otherwise, assume the agent already holds the key (flag 1).
    """
    for row in grid:
        tokens = row.split()
        for token in tokens:
            if token == 'k':
                return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True, help='Path to MDP file (from encoder)')
    parser.add_argument('--value-policy', required=True, help='Path to value-policy file (planner output)')
    parser.add_argument('--gridworld', required=True, help='Path to gridworld test file')
    args = parser.parse_args()
    
    # Read the value-policy file (each line: "value    policy").
    with open(args.value_policy, 'r') as f:
        vp_lines = f.read().splitlines()
    policy_list = []
    for line in vp_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            policy_list.append(int(parts[1]))
    
    # Parse test cases.
    test_cases = parse_grid_test(args.gridworld)
    actions = []
    for case in test_cases:
        grid = case  # each test case is a list of grid rows.
        agent_info = find_agent(grid)
        if agent_info is None:
            actions.append("0")
            continue
        i, j, orient = agent_info
        # Determine key flag: 0 if key is present, 1 otherwise.
        key_flag = 0 if has_key_in_grid(grid) else 1
        free_cells = build_free_cells_from_test(grid)
        # Compute state id using same ordering as in encoder:
        # state = ((cell_index * 4) + orient) * 2 + key_flag.
        cell_index = free_cells[(i, j)]
        state = ((cell_index * 4) + orient) * 2 + key_flag
        if state < len(policy_list):
            optimal_action = policy_list[state]
        else:
            optimal_action = 0
        actions.append(str(optimal_action))
    
    # Print the optimal actions for all test cases (space-separated).
    print(" ".join(actions))

if __name__ == "__main__":
    main()
