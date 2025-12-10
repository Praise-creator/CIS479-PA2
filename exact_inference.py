from network_definition import NODES
from bayes_network import get_probability, get_all_parent_values
import itertools

def enumerate_all(variables, evidence):
    """
    Core enumeration algorithm,
    args:
        variables: list of all variable names in topological order
        evidence: dictionary of variable assignments {variable: value}
        
    returns:
        Sum of probabilities over all assignments consistent with evidence
    """
    if not variables:
        return 1.0
    
    # Get the first variable
    var = variables[0]
    remaining = variables[1:]
    
    if var in evidence:
        # If variable is in evidence, use that value
        val = evidence[var]
        parent_values = get_all_parent_values(var, evidence)
        prob = get_probability(var, val, parent_values)
        return prob * enumerate_all(remaining, evidence)
    else:
        # If variable is not in evidence, sum over both possible values
        total = 0.0
        for val in [True, False]:
            new_evidence = evidence.copy()
            new_evidence[var] = val
            parent_values = get_all_parent_values(var, new_evidence)
            prob = get_probability(var, val, parent_values)
            total += prob * enumerate_all(remaining, new_evidence)
        return total


def exact_inference(query_vars, evidence):
    """
    Compute P(query_vars | evidence) using enumeration.
    
    Args:
        query_vars: List of query variable names (e.g., ['J'] or ['M', 'A'])
        evidence: Dictionary of evidence {variable: value} (e.g., {'A': True, 'B': False})
    
    Returns:
        Dictionary mapping query variable assignments to probabilities
        Example: {(True,): 0.9} for single variable or {(True, True): 0.5, ...} for multiple
    """
    query_combinations = list(itertools.product([True, False], repeat=len(query_vars)))
    
    unnormalized = {}
    
    for combination in query_combinations:
        extended_evidence = evidence.copy()
        for i, var in enumerate(query_vars):
            extended_evidence[var] = combination[i]
        
        # Enumerate over all variables in topological order
        prob = enumerate_all(NODES, extended_evidence)
        unnormalized[combination] = prob
    
    # Normalize the probabilities
    total = sum(unnormalized.values())
    normalized = {k: v / total for k, v in unnormalized.items()}
    
    return normalized


def query_exact(query_vars, evidence):
    """
    Run exact inference and return results.
    
    Args:
        query_vars: List of query variables
        evidence: Dictionary of evidence
    
    Returns:
        Dictionary of probabilities for each query assignment
    """
    return exact_inference(query_vars, evidence)