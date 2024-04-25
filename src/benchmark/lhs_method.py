import numpy as np
from pyDOE2 import lhs

def generate_lhs_samples(dimension_lengths, num_samples):
    LHS_sample = lhs(n=len(dimension_lengths), samples=num_samples, criterion='maximin')  # Using maximin criterion
    scaled_samples = np.floor(LHS_sample * np.array(dimension_lengths)).astype(int)
    return scaled_samples