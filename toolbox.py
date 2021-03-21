import numpy as np
from random import randrange

def discreteProb(p):
        # Draw a random number using probability table p (column vector)
        # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1 
        # and the components p(j) are nonnegative. 
        # To generate a random sample of size m from this distribution,
        # imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
        # Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
        # return the value j. Repeat m times.
        r = np.random.random()
        cumprob = np.hstack((np.zeros(1), np.cumsum(p)))
        sample = -1
        for j in range(p.size):
            if (r > cumprob[j]) & (r <= cumprob[j+1]):
                sample = j
                break
        return sample


def random_policy(mdp):
    # Returns a random policy given an mdp
    # Inputs :
    # - mdp : the mdp
    # Output :
    # - pol : the policy

    rand = np.random
    pol = np.zeros(mdp.nb_states, dtype=np.int16)
    for x in range(mdp.nb_states):
        pol[x] = rand.choice(mdp.action_space.actions)
    return pol