import random
import itertools
from network_definition import NODES, PARENTS
from bayes_network import get_probability, get_all_parent_values


def generate_prior_sample():
    """
    Generate one sample from the prior distribution (no evidence).
    Sample each variable in topological order based on its parents.
    
    Returns:
        Dictionary {variable: value} representing one complete sample
    """
    sample = {}
    
    for node in NODES:
        # Get parent values from what we've already sampled
        parent_values = get_all_parent_values(node, sample)
        
        # Get probability of this node being True given its parents
        prob_true = get_probability(node, True, parent_values)
        
        # Sample: generate random number and compare to probability
        sample[node] = random.random() < prob_true
    
    return sample


def prior_sampling(query_vars, evidence, num_samples):
    """
    Approximate P(query_vars | evidence) using prior sampling.
    Generate samples and count those matching both evidence and query.
    
    Args:
        query_vars: List of query variable names
        evidence: Dictionary of evidence {variable: value}
        num_samples: Number of samples to generate
    
    Returns:
        Dictionary mapping query assignments to probabilities
    """
    # Count matches for each query combination
    counts = {}
    total_matching_evidence = 0
    
    for _ in range(num_samples):
        sample = generate_prior_sample()
        
        # Check if sample matches evidence
        matches_evidence = all(sample[var] == val for var, val in evidence.items())
        
        if matches_evidence:
            total_matching_evidence += 1
            
            # Extract query values from this sample
            query_values = tuple(sample[var] for var in query_vars)
            counts[query_values] = counts.get(query_values, 0) + 1
    
    # Convert counts to probabilities
    if total_matching_evidence == 0:
        # No samples matched evidence - return uniform distribution
        num_combinations = 2 ** len(query_vars)
        return {tuple([True if i & (1 << j) else False for j in range(len(query_vars))]): 
                1.0 / num_combinations for i in range(num_combinations)}
    
    probabilities = {k: v / total_matching_evidence for k, v in counts.items()}
    
    # Fill in missing combinations with 0 probability
    import itertools
    for combination in itertools.product([True, False], repeat=len(query_vars)):
        if combination not in probabilities:
            probabilities[combination] = 0.0
    
    return probabilities


def rejection_sampling(query_vars, evidence, num_samples):
    """
    Approximate P(query_vars | evidence) using rejection sampling.
    Generate samples and reject those that don't match evidence.
    
    Args:
        query_vars: List of query variable names
        evidence: Dictionary of evidence
        num_samples: Number of samples to generate (before rejection)
    
    Returns:
        Dictionary mapping query assignments to probabilities
    """
    counts = {}
    total_accepted = 0
    
    samples_generated = 0
    while samples_generated < num_samples:
        sample = generate_prior_sample()
        samples_generated += 1
        
        # Check if sample matches evidence
        matches_evidence = all(sample[var] == val for var, val in evidence.items())
        
        if matches_evidence:
            # Accept this sample
            total_accepted += 1
            
            # Extract query values
            query_values = tuple(sample[var] for var in query_vars)
            counts[query_values] = counts.get(query_values, 0) + 1
    
    # Convert counts to probabilities
    if total_accepted == 0:
        num_combinations = 2 ** len(query_vars)
        return {tuple([True if i & (1 << j) else False for j in range(len(query_vars))]): 
                1.0 / num_combinations for i in range(num_combinations)}
    
    probabilities = {k: v / total_accepted for k, v in counts.items()}
    
    # Fill in missing combinations
    import itertools
    for combination in itertools.product([True, False], repeat=len(query_vars)):
        if combination not in probabilities:
            probabilities[combination] = 0.0
    
    return probabilities


def weighted_sample(evidence):
    """
    Generate one weighted sample for likelihood weighting.
    Evidence variables are fixed, others are sampled.
    
    Args:
        evidence: Dictionary of evidence {variable: value}
    
    Returns:
        Tuple of (sample, weight) where sample is a dict and weight is a float
    """
    sample = {}
    weight = 1.0
    
    for node in NODES:
        parent_values = get_all_parent_values(node, sample)
        
        if node in evidence:
            # Evidence variable: fix its value and update weight
            sample[node] = evidence[node]
            # Weight is multiplied by P(node=evidence_value | parents)
            weight *= get_probability(node, evidence[node], parent_values)
        else:
            # Non-evidence variable: sample as usual
            prob_true = get_probability(node, True, parent_values)
            sample[node] = random.random() < prob_true
    
    return sample, weight


def likelihood_weighting(query_vars, evidence, num_samples):
    """
    Approximate P(query_vars | evidence) using likelihood weighting.
    Generate weighted samples where evidence is fixed.
    
    Args:
        query_vars: List of query variable names
        evidence: Dictionary of evidence
        num_samples: Number of weighted samples to generate
    
    Returns:
        Dictionary mapping query assignments to probabilities
    """
    weighted_counts = {}
    
    for _ in range(num_samples):
        sample, weight = weighted_sample(evidence)
        
        # Extract query values
        query_values = tuple(sample[var] for var in query_vars)
        weighted_counts[query_values] = weighted_counts.get(query_values, 0) + weight
    
    # Normalize by total weight
    total_weight = sum(weighted_counts.values())
    
    if total_weight == 0:
        num_combinations = 2 ** len(query_vars)
        return {tuple([True if i & (1 << j) else False for j in range(len(query_vars))]): 
                1.0 / num_combinations for i in range(num_combinations)}
    
    probabilities = {k: v / total_weight for k, v in weighted_counts.items()}
    
    # Fill in missing combinations
    import itertools
    for combination in itertools.product([True, False], repeat=len(query_vars)):
        if combination not in probabilities:
            probabilities[combination] = 0.0
    
    return probabilities