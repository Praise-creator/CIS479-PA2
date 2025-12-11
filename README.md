##Files
- `network_definition.py` - Network structure and CPT probability tables
- `bayes_network.py` - Helper functions for probability calculations
- `exact_inference.py` - Exact inference by enumeration
- `sampling_inference.py` - Prior Sampling, Rejection Sampling, Likelihood Weighting
- `main.py` - Main program with query interface
- `inference_report.pdf` - Report of the 3 sampling methods
- `ByesNetwork.png` - screenshot of the network from the textbook


To run the main file and generate the comparison of the 3 sampling methods, do python main.py analyze

if you want to run specific queries, run python main.py and enter query in the required format 
"""
Enter queries in format: [<N1,V1><N2,V2>][Q1,Q2]
Nodes: A (Alarm), B (Burglary), E (Earthquake), J (John), M (Mary)
Values: t (true), f (false)
Examples:
  [<A,t><B,f>][J]      # Query John given Alarm=true, Burglary=false
  [<E,t><J,t>][M,A]    # Query Mary and Alarm given Earthquake and John are true
  Type 'quit' to exit or 'analyze' to run the three cases

  """



  