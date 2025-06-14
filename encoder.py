#!/usr/bin/env python3
import sys
import argparse

def parse_grid(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().splitlines()
    # Each line is assumed to have space-separated characters.
    grid = [line.split() for line in lines]
    return grid

def build_free_cells(grid):
    free_cells = {}
    counter = 0
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell != 'W':  # every non-wall is free
                free_cells[(i, j)] = counter
                counter += 1
    return free_cells

def in_bounds(grid, i, j):
    return 0 <= i < len(grid) and 0 <= j < len(grid[0])

def traversable(grid, i, j, has_key):
    if not in_bounds(grid, i, j):
        return False
    cell = grid[i][j]
    if cell == 'W':
        return False
    # The door cell is traversable only when key is held.
    if cell == 'd' and not has_key:
        return False
    return True

def get_sliding_outcomes(grid, pos, orient, has_key):
    """
    For action 0 (move forward) the agent slides 1, 2 or 3 steps with
    p(1)=0.5, p(2)=0.3, p(3)=0.2. If the agent is blocked (by wall or door
    when key is not held), then all probability mass for moves beyond the 
    maximum possible is added to the outcome corresponding to the maximum.
    """
    i, j = pos
    # Directions: 0: up, 1: right, 2: down, 3: left.
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    di, dj = dirs[orient]
    max_possible = 0
    valid_positions = {}
    for d in [1, 2, 3]:
        ni = i + d * di
        nj = j + d * dj
        if traversable(grid, ni, nj, has_key):
            max_possible = d
            valid_positions[d] = (ni, nj)
        else:
            break
    outcomes = []
    if max_possible == 0:
        # Cannot move; remain in place.
        outcomes.append((pos, 1.0))
    else:
        if max_possible == 1:
            outcomes.append((valid_positions[1], 1.0))
        elif max_possible == 2:
            outcomes.append((valid_positions[1], 0.5))
            outcomes.append((valid_positions[2], 0.5))  # (0.3+0.2 combined)
        elif max_possible == 3:
            outcomes.append((valid_positions[1], 0.5))
            outcomes.append((valid_positions[2], 0.3))
            outcomes.append((valid_positions[3], 0.2))
    return outcomes

def encode(grid):
    free_cells = build_free_cells(grid)
    num_free = len(free_cells)
    numStates = num_free * 4 * 2  # each free cell: 4 orientations, 2 key statuses.
    numActions = 4
    lines = []
    lines.append("numStates " + str(numStates))
    lines.append("numActions " + str(numActions))
    
    # Helper: compute state id from cell position, orientation and key flag.
    def state_id(i, j, orient, key):
        return ((free_cells[(i, j)] * 4) + orient) * 2 + key
    
    # Identify goal cells (terminal states).
    goal_positions = set()
    for (i, j) in free_cells:
        if grid[i][j] == 'g':
            goal_positions.add((i, j))
    
    # For every state (free cell, orientation, key flag) add transitions.
    for (i, j) in free_cells:
        for orient in range(4):
            for key in [0, 1]:
                s = state_id(i, j, orient, key)
                # Terminal states (when in a goal cell) have no outgoing transitions.
                if (i, j) in goal_positions:
                    continue
                for a in range(4):
                    # All actions incur a cost of -1.
                    if a == 0:
                        # Move forward: determine sliding outcomes.
                        outcomes = get_sliding_outcomes(grid, (i, j), orient, (key == 1))
                        for (ni, nj), prob in outcomes:
                            new_key = key
                            # If the cell has the key and agent didn’t already have it, pick it up.
                            if grid[ni][nj] == 'k' and key == 0:
                                new_key = 1
                            s_next = state_id(ni, nj, orient, new_key)
                            lines.append("transition {} {} {} {} {}".format(s, a, s_next, -1, prob))
                    elif a == 1:
                        # Turn left: 0.9 for left, 0.1 for turn around.
                        new_orient1 = (orient - 1) % 4
                        new_orient2 = (orient + 2) % 4
                        s_next1 = state_id(i, j, new_orient1, key)
                        s_next2 = state_id(i, j, new_orient2, key)
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next1, -1, 0.9))
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next2, -1, 0.1))
                    elif a == 2:
                        # Turn right: 0.9 for right, 0.1 for turn around.
                        new_orient1 = (orient + 1) % 4
                        new_orient2 = (orient + 2) % 4
                        s_next1 = state_id(i, j, new_orient1, key)
                        s_next2 = state_id(i, j, new_orient2, key)
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next1, -1, 0.9))
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next2, -1, 0.1))
                    elif a == 3:
                        # Turn around: 0.8 for turn around, 0.1 for left, 0.1 for right.
                        new_orient1 = (orient + 2) % 4
                        new_orient2 = (orient - 1) % 4
                        new_orient3 = (orient + 1) % 4
                        s_next1 = state_id(i, j, new_orient1, key)
                        s_next2 = state_id(i, j, new_orient2, key)
                        s_next3 = state_id(i, j, new_orient3, key)
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next1, -1, 0.8))
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next2, -1, 0.1))
                        lines.append("transition {} {} {} {} {}".format(s, a, s_next3, -1, 0.1))
    # List terminal states (all states with a goal cell).
    term_states = []
    for (i, j) in free_cells:
        if (i, j) in goal_positions:
            for orient in range(4):
                for key in [0, 1]:
                    term_states.append(state_id(i, j, orient, key))
    lines.append("end " + " ".join(map(str, term_states)))
    lines.append("mdptype episodic")
    lines.append("discount 1.0")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridworld', required=True, help='Path to gridworld file')
    args = parser.parse_args()
    grid = parse_grid(args.gridworld)
    mdp_str = encode(grid)
    print(mdp_str)

if __name__ == "__main__":
    main()
