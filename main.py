import sys 
import random
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

from network_definition import *
from bayes_network import get_probability, get_all_parent_values
from exact_inference import query_exact
from sampling_methods import prior_sampling, rejection_sampling, likelihood_weighting

def parse_input(input_str):
    input_str = input_str.strip()

    if '[]' not in input_str:
        raise ValueError("Input format should be [evidence][query]")
    
    evidence_end = input_str.find('][')
    evidence_str = input_str[1:evidence_end]
    query_str = input_str[evidence_end+2:-1]

    evidence = {}
    if evidence_str.strip():
        # Split by '><' but handle edge cases
        items = evidence_str.replace('><', '|').split('|')
        for item in items:
            item = item.strip()
            if not item:
                continue
            # Remove < and >
            if item.startswith('<'):
                item = item[1:]
            if item.endswith('>'):
                item = item[:-1]
            
            # Split node and value
            parts = item.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid evidence item: {item}")
            
            node = parts[0].strip()
            value_str = parts[1].strip()
            
            # Convert value to boolean
            if value_str.lower() in ['t', 'true']:
                value = True
            elif value_str.lower() in ['f', 'false']:
                value = False
            else:
                raise ValueError(f"Invalid value: {value_str}")
            
            # Check node is valid
            if node not in ['A', 'B', 'E', 'J', 'M']:
                raise ValueError(f"Invalid node: {node}")
            
            evidence[node] = value
    
    # Parse query
    query_vars = []
    if query_str.strip():
        items = query_str.split(',')
        for item in items:
            var = item.strip()
            if var in ['A', 'B', 'E', 'J', 'M']:
                query_vars.append(var)
            else:
                raise ValueError(f"Invalid query variable: {var}")
    
    return evidence, query_vars