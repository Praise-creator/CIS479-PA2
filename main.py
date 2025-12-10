import sys 
import random
import itertools
from collections import defaultdict

from network_definition import *
from bayes_network import get_probability, get_all_parent_values
from exact_inference import query_exact
from sampling_inference import prior_sampling, rejection_sampling, likelihood_weighting

def parse_input(input_str):
    input_str = input_str.strip()

    if not input_str.startswith('['):
        raise ValueError("Input must start with '['")
    
    depth = 0
    evidence_end = -1
    for i, char in enumerate(input_str):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and evidence_end == -1:
                evidence_end = i
                break
    
    if evidence_end == -1 or evidence_end + 1 >= len(input_str) or input_str[evidence_end+1] != '[':
        raise ValueError("Invalid format. Expected [evidence][query]")
    
    evidence_str = input_str[1:evidence_end]
    query_str = input_str[evidence_end+2:-1]

    evidence = {}
    if evidence_str.strip():
        evidence_str = evidence_str.replace(" ", "")
    
        # Split into evidence items
        items = []
        start = 0
        while start < len(evidence_str):
            if evidence_str[start] == '<':
                end = evidence_str.find('>', start)
                if end == -1:
                    raise ValueError("Unclosed '<' in evidence")
                items.append(evidence_str[start+1:end])
                start = end + 1
            else:
                start += 1
        
        for item in items:
            parts = item.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid evidence item: {item}")
            
            node = parts[0]
            value_str = parts[1].lower()
            
            if value_str == 't':
                value = True
            elif value_str == 'f':
                value = False
            else:
                raise ValueError(f"Invalid value: {value_str}. Use 't' or 'f'.")
            
            if node not in ['A', 'B', 'E', 'J', 'M']:
                raise ValueError(f"Invalid node: {node}")
            
            evidence[node] = value
    
    # Parse query
    query_vars = []
    if query_str.strip():
        query_str = query_str.replace(" ", "")
        items = query_str.split(',')
        for item in items:
            if item in ['A', 'B', 'E', 'J', 'M']:
                query_vars.append(item)
            else:
                raise ValueError(f"Invalid query variable: {item}")
    
    return evidence, query_vars


def format_output(query_vars, probabilities):
    """
    Format output as [<NQ1,P1><NQ2,P2>...]
    """
    if not query_vars:
        return "[]"
    
    if len(query_vars) == 1:
        # Single query variable
        true_prob = probabilities.get((True,), 0)
        return f"[<{query_vars[0]},{true_prob:.4f}>]"
    else:
        # Joint query - probability that all are True
        all_true = tuple([True] * len(query_vars))
        joint_prob = probabilities.get(all_true, 0)
        nodes_str = ','.join(query_vars)
        return f"[<{nodes_str},{joint_prob:.6f}>]"


def run_sampling_trials(query_vars, evidence, num_samples, num_trials=10):
    """
    Run sampling methods multiple times and return average results.
    """
    # For joint queries, we want probability that all are True
    target_key = tuple([True] * len(query_vars))
    
    prior_results = []
    rejection_results = []
    likelihood_results = []
    
    for _ in range(num_trials):
        # Prior sampling
        prior_probs = prior_sampling(query_vars, evidence, num_samples)
        prior_results.append(prior_probs.get(target_key, 0))
        
        # Rejection sampling
        reject_probs = rejection_sampling(query_vars, evidence, num_samples)
        rejection_results.append(reject_probs.get(target_key, 0))
        
        # Likelihood weighting
        lw_probs = likelihood_weighting(query_vars, evidence, num_samples)
        likelihood_results.append(lw_probs.get(target_key, 0))
    
    # Calculate averages
    avg_prior = sum(prior_results) / len(prior_results) if prior_results else 0
    avg_rejection = sum(rejection_results) / len(rejection_results) if rejection_results else 0
    avg_likelihood = sum(likelihood_results) / len(likelihood_results) if likelihood_results else 0
    
    return {
        'prior': avg_prior,
        'rejection': avg_rejection,
        'likelihood': avg_likelihood,
        'prior_all': prior_results,
        'rejection_all': rejection_results,
        'likelihood_all': likelihood_results
    }


def analyze_specific_cases():
    """
    Analyze the three specific cases mentioned in the assignment
    """
    print("\n" + "="*80)
    print("ANALYSIS OF THREE SPECIFIC CASES")
    print("Each method run 10 times and averaged for each sample size")
    print("="*80)
    
    cases = [
        {
            "name": "Case 1: Alarm is false, infer Burglary and JohnCalls being true",
            "evidence": {"A": False},
            "query": ["B", "J"],
            "description": "P(B=true, J=true | A=false)"
        },
        {
            "name": "Case 2: JohnCalls is true, Earthquake is false, infer Burglary and MaryCalls being true",
            "evidence": {"J": True, "E": False},
            "query": ["B", "M"],
            "description": "P(B=true, M=true | J=true, E=false)"
        },
        {
            "name": "Case 3: MaryCalls is true and JohnCalls is false, infer Burglary and Earthquake being true",
            "evidence": {"M": True, "J": False},
            "query": ["B", "E"],
            "description": "P(B=true, E=true | M=true, J=false)"
        }
    ]
    
    sample_sizes = [1, 10, 100, 1000, 10000]
    
    for case_idx, case in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"CASE {case_idx}: {case['name']}")
        print(f"Query: {case['description']}")
        print(f"{'='*60}")
        
        # Get exact probability first
        try:
            exact_result = query_exact(case['query'], case['evidence'])
            all_true_key = tuple([True] * len(case['query']))
            exact_prob = exact_result.get(all_true_key, 0)
            print(f"Exact probability: {exact_prob:.8f}")
        except Exception as e:
            print(f"Error in exact inference: {e}")
            exact_prob = 0
        
        # Run sampling methods for different sample sizes
        print(f"\n{'Samples':<10} {'Prior':<15} {'Rejection':<15} {'Likelihood':<15}")
        print(f"{'-'*55}")
        
        for n in sample_sizes:
            # Run each method 10 times and average
            results = run_sampling_trials(case['query'], case['evidence'], n, 10)
            
            print(f"{n:<10} {results['prior']:<15.8f} {results['rejection']:<15.8f} {results['likelihood']:<15.8f}")
        
        # Add exact result row
        print(f"{'Exact':<10} {exact_prob:<15.8f} {exact_prob:<15.8f} {exact_prob:<15.8f}")


def interactive_mode():
    """
    Interactive mode for testing queries
    """
    print("\n" + "="*80)
    print("INTERACTIVE INFERENCE MODE")
    print("="*80)
    print("Enter queries in format: [<N1,V1><N2,V2>][Q1,Q2]")
    print("Nodes: A (Alarm), B (Burglary), E (Earthquake), J (John), M (Mary)")
    print("Values: t (true), f (false)")
    print("Examples:")
    print("  [<A,t><B,f>][J]      # Query John given Alarm=true, Burglary=false")
    print("  [<E,t><J,t>][M,A]    # Query Mary and Alarm given Earthquake and John are true")
    print("  Type 'quit' to exit or 'analyze' to run the three cases")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nEnter query: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'analyze':
                analyze_specific_cases()
                continue
            
            # Parse input
            evidence, query_vars = parse_input(user_input)
            
            print(f"\nEvidence: {evidence}")
            print(f"Query variables: {query_vars}")
            
            if not query_vars:
                print("Error: No query variables specified")
                continue
            
            # Get exact inference
            try:
                exact_result = query_exact(query_vars, evidence)
                exact_output = format_output(query_vars, exact_result)
                print(f"Exact Inference: {exact_output}")
                
                # Show all probabilities for debugging
                if len(query_vars) == 1:
                    true_prob = exact_result.get((True,), 0)
                    false_prob = exact_result.get((False,), 0)
                    print(f"  P({query_vars[0]}=True) = {true_prob:.6f}, P({query_vars[0]}=False) = {false_prob:.6f}")
            except Exception as e:
                print(f"Error in exact inference: {e}")
                continue
            
            # Ask for sampling
            sampling_choice = input("\nRun sampling methods? (y/n): ").strip().lower()
            if sampling_choice == 'y':
                try:
                    num_samples = int(input("Number of samples (default 1000): ") or "1000")
                    
                    # Run sampling methods
                    print(f"\nRunning with {num_samples} samples...")
                    
                    # Prior sampling
                    prior_result = prior_sampling(query_vars, evidence, num_samples)
                    prior_output = format_output(query_vars, prior_result)
                    print(f"Prior Sampling:    {prior_output}")
                    
                    # Rejection sampling
                    reject_result = rejection_sampling(query_vars, evidence, num_samples)
                    reject_output = format_output(query_vars, reject_result)
                    print(f"Rejection Sampling: {reject_output}")
                    
                    # Likelihood weighting
                    lw_result = likelihood_weighting(query_vars, evidence, num_samples)
                    lw_output = format_output(query_vars, lw_result)
                    print(f"Likelihood Weighting: {lw_output}")
                    
                except ValueError:
                    print("Invalid number of samples")
                except Exception as e:
                    print(f"Error in sampling: {e}")
            
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


def main():
    """
    Main entry point
    """
    print("Bayesian Network Inference System")
    print("Network: Burglary-Earthquake-Alarm (from Russell & Norvig 4th ed, Figure 13.2)")
    print("CPT values from textbook:")
    print("  P(B) = 0.001, P(E) = 0.002")
    print("  P(A|B,E): B=t,E=t:0.95, B=t,E=f:0.94, B=f,E=t:0.29, B=f,E=f:0.001")
    print("  P(J|A): A=t:0.90, A=f:0.05")
    print("  P(M|A): A=t:0.70, A=f:0.01")
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == 'analyze':
            analyze_specific_cases()
        elif sys.argv[1] == 'test':
            test_queries()
        else:
            # Single query from command line
            input_str = ' '.join(sys.argv[1:])
            try:
                evidence, query_vars = parse_input(input_str)
                print(f"Evidence: {evidence}")
                print(f"Query: {query_vars}")
                
                # Get exact result
                exact_result = query_exact(query_vars, evidence)
                output = format_output(query_vars, exact_result)
                print(f"Result: {output}")
                
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Interactive mode
        interactive_mode()


def test_queries():
    """Test some example queries"""
    test_cases = [
        ("[<A,t><B,f>][J]", "P(J | A=true, B=false)"),
        ("[<E,t><J,t>][M,A]", "P(M,A | E=true, J=true)"),
        ("[][B]", "Prior P(B)"),
        ("[<J,t>][A]", "P(A | J=true)"),
    ]
    
    print("\nTest Queries:")
    print("="*60)
    
    for input_str, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {input_str}")
        
        try:
            evidence, query_vars = parse_input(input_str)
            print(f"Evidence: {evidence}")
            print(f"Query: {query_vars}")
            
            # Get exact result
            exact_result = query_exact(query_vars, evidence)
            output = format_output(query_vars, exact_result)
            print(f"Result: {output}")
            
            # Show distribution for single queries
            if len(query_vars) == 1:
                true_prob = exact_result.get((True,), 0)
                false_prob = exact_result.get((False,), 0)
                print(f"Distribution: P(True)={true_prob:.6f}, P(False)={false_prob:.6f}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-"*60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()