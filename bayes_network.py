from network_definition import *

def get_probability(node, value, parent_values):
    if node == 'B':
        return P_BURGLARY[value]
    elif node == 'E':
        return P_EARTHQUAKE[value]
    elif node == 'A':
        b_val = parent_values['B']
        e_val = parent_values['E']
        prob = P_ALARM[(b_val, e_val)]
        return prob if value else 1 - prob
    elif node == 'J':
        a_val = parent_values['A']
        prob = P_JOHN[a_val]
        return prob if value else 1 - prob
        return P_JOHN[parent_values]
    elif node == 'M':
        a_val = parent_values['A']
        prob = P_MARY[a_val]
        return prob if value else 1 - prob
    else:
        raise ValueError("Unknown node")
    

def get_all_parent_values(node, assignment):
    parent_values = {}
    for parent in PARENTS[node]:
        parent_values[parent] = assignment[parent]
    return parent_values


def print_probability_distribution(query_vars, probabilities):
    for values, prob in probabilities.items():
        assignment_str = ', '.join(f"{var}={val}" for var, val in zip(query_vars, values))
        print(f"P({assignment_str}) = {prob:.6f}")