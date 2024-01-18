import os
import sys
import math
import random
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


import matplotlib.pyplot as plt



min_sad = tfd.JointDistributionNamed(
    dict(
        kappa_min = tfd.Uniform(low=0,high=0.99),
        gamma_min = lambda kappa_min: tfd.Uniform(low=0.0,high=1-kappa_min),
        #s_min = tfd.Uniform(low=0,high=1.0),
        s_min = tfd.FiniteDiscrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], probs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        kappa_sad = tfd.Uniform(low=0,high=1.0),
        gamma_sad = lambda kappa_sad: tfd.Uniform(low=1-kappa_sad,high=1.2),
        #s_sad = tfd.Uniform(low=0,high=1.0),
        s_sad = tfd.FiniteDiscrete([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], probs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        vel = tfd.Uniform(low=5000,high=15000), # km/s
        mass = tfd.Uniform(0.1,2) # solar masses
        )
)



my_sample = min_sad.sample(1000)
sample = {}
for key,tensor in my_sample.items():
    sample[key] = tensor.numpy()

print(min_sad.resolve_graph())


fig,ax = plt.subplots(1)
ax.scatter(sample['kappa_sad'],sample['gamma_sad'])
plt.savefig('kg.png')
    
