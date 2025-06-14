import argparse
import numpy as np
import pulp

def parse_mdp(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    S, A = 0, 0
    transitions = {}
    terminal_states = set()
    mdptype, gamma = None, None
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'numStates':
            S = int(parts[1])
        elif parts[0] == 'numActions':
            A = int(parts[1])
        elif parts[0] == 'end':
            terminal_states = set(map(int, parts[1:]))
        elif parts[0] == 'transition':
            s1, a, s2, r, p = map(float, parts[1:])
            s1, a, s2 = int(s1), int(a), int(s2)
            if (s1, a) not in transitions:
                transitions[(s1, a)] = []
            transitions[(s1, a)].append((p, s2, r))
        elif parts[0] == 'mdptype':
            mdptype = parts[1]
        elif parts[0] == 'discount':
            gamma = float(parts[1])
    
    return S, A, transitions, terminal_states, mdptype, gamma

def policy_evaluation_exact(S, transitions, terminal_states, policy):
    A_mat = np.eye(S)
    b = np.zeros(S)
    for s in range(S):
        if s in terminal_states:
            b[s] = 0
            continue
        # Each action incurs a cost of -1.
        b[s] = -1
        for p, s_next, r in transitions.get((s, policy[s]), []):
            A_mat[s, s_next] -= p
    V = np.linalg.solve(A_mat, b)
    return V

def  Howard_policy_iteration_exact(S, A, transitions, terminal_states, gamma):
    policy = np.zeros(S, dtype=int)
    while True:
        try:
            V = policy_evaluation_exact(S, transitions, terminal_states, policy)
        except np.linalg.LinAlgError:
            # If the linear system is singular, fall back to LP.
            return linear_programming(S, A, transitions, terminal_states, gamma)
        stable = True
        for s in range(S):
            if s in terminal_states:
                continue
            best_a, best_value = None, -np.inf
            for a in range(A):
                if (s, a) in transitions:
                    value = sum(p * (r + gamma * V[s_next]) for p, s_next, r in transitions[(s, a)])
                    if value > best_value:
                        best_a, best_value = a, value
            if best_a is not None and best_a != policy[s]:
                policy[s] = best_a
                stable = False
        if stable:
            break
    return V, policy

def linear_programming(S, A, transitions, terminal_states, gamma):
    prob = pulp.LpProblem("MDP_LP", pulp.LpMinimize)
    V_vars = [pulp.LpVariable(f'V{s}', lowBound=None) for s in range(S)]
    prob += pulp.lpSum(V_vars)
    for s in range(S):
        if s in terminal_states:
            prob += V_vars[s] == 0
            continue
        for a in range(A):
            if (s, a) in transitions:
                prob += V_vars[s] >= pulp.lpSum(p * (r + gamma * V_vars[s_next])
                                                for p, s_next, r in transitions[(s, a)])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    V_values = [pulp.value(V_vars[s]) if pulp.value(V_vars[s]) is not None else 0 for s in range(S)]
    policy = np.zeros(S, dtype=int)
    for s in range(S):
        if s in terminal_states:
            policy[s] = 0
            continue
        best_a, best_value = None, -np.inf
        for a in range(A):
            if (s, a) in transitions:
                value = sum(p * (r + gamma * V_values[s_next])
                            for p, s_next, r in transitions[(s, a)])
                if value > best_value:
                    best_a, best_value = a, value
        policy[s] = best_a if best_a is not None else 0
    return V_values, policy

def policy_evaluation(S, A, transitions, terminal_states, gamma, policy):
    V = np.zeros(S)
    while True:
        V_new = np.zeros(S)
        for s in range(S):
            if s in terminal_states:
                continue
            a = policy[s]
            if (s, a) in transitions:
                V_new[s] = sum(p * (r + gamma * V[s_next])
                               for p, s_next, r in transitions[(s, a)])
        if np.max(np.abs(V_new - V)) < 1e-8:
            break
        V = V_new
    return V

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True, help='Path to MDP file')
    parser.add_argument('--algorithm', choices=['hpi', 'lp'], default='hpi', help='Algorithm to use')
    parser.add_argument('--policy', help='Policy file for evaluation')
    args = parser.parse_args()
    
    S, A, transitions, terminal_states, mdptype, gamma = parse_mdp(args.mdp)
    
    if args.policy:
        with open(args.policy, 'r') as f:
            policy = [int(line.strip()) for line in f.readlines()]
        V = policy_evaluation(S, A, transitions, terminal_states, gamma, policy)
        for s in range(S):
            print(f"{V[s]:.8f}\t{policy[s]}")
        return
    
    # For episodic MDPs (as in gridworlds), use LP to avoid singular system issues.
    if mdptype == 'episodic':
        V, policy = linear_programming(S, A, transitions, terminal_states, gamma)
    else:
        if args.algorithm == 'hpi':
            V, policy = Howard_policy_iteration_exact(S, A, transitions, terminal_states, gamma)
        else:
            V, policy = linear_programming(S, A, transitions, terminal_states, gamma)
    
    for s in range(S):
        print(f"{V[s]:.8f}\t{policy[s]}")

if __name__ == "__main__":
    main()
