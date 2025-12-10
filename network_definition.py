# Bayesian Network Definition for Burglary-Earthquake Scenario

P_BURGLARY = {
    True: 0.001,
    False: 0.999
}

P_EARTHQUAKE = {
    True: 0.002,
    False: 0.998
}

# P(Alarm | Burglary, Earthquake)
P_ALARM = {
    (True, True): 0.95,    # P(A=true | B=true, E=true)
    (True, False): 0.94,   # P(A=true | B=true, E=false)
    (False, True): 0.29,   # P(A=true | B=false, E=true)
    (False, False): 0.001  # P(A=true | B=false, E=false)
}

# P(JohnCalls | Alarm)
P_JOHN = {
    True: 0.90,   # P(J=true | A=true)
    False: 0.05   # P(J=true | A=false)
}

# P(MaryCalls | Alarm)
P_MARY = {
    True: 0.70,   # P(M=true | A=true)
    False: 0.01   # P(M=true | A=false)
}

# parent relationships
PARENTS = {
    'B': [],
    'E': [],
    'A': ['B', 'E'],
    'J': ['A'],
    'M': ['A']
}

# All nodes in topological order
NODES = ['B', 'E', 'A', 'J', 'M']