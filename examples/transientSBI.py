#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import modules

from transient_smash.models.model import SinusoidalModelWithNoise
from transient_smash.sbi_interface.sbi_interface import NLESBI
import matplotlib.pyplot as plt
import numpy as np
from nautilus import Sampler, Prior
from sbi.inference import likelihood_estimator_based_potential
import torch
from sbi.analysis import pairplot


# In[2]:


n_simulations = 1000 # We want 1000 simulations
n_observations = 1000 # We want 1000 obs in time
x = np.linspace(0,10,n_observations) 
noisy_params = [1,1,1,1,0,1] # set 6 params by specifying their mean values


# In[3]:


true_model = SinusoidalModelWithNoise() # instantiate model
observations = true_model.evaluate(x,noisy_params) # create observations

# Plot observations
plt.plot(x,observations)
plt.ylabel('Observation Value')
plt.xlabel('Observation Number')
plt.tight_layout()
plt.show()
plt.close()


# Plot observations
# plt.plot(x,observations)
# plt.ylabel('Observation Value')
# plt.xlabel('Observation Number')
# plt.tight_layout()
# plt.show()
# plt.close()

# In[4]:


test_model_1 = SinusoidalModelWithNoise() # define test model
priors_dict = {"amplitude": ("uniform", 0.0, 2.0),
               "frequency": ("uniform", 0.0, 2.0),
               "phase": ("uniform", 0.0, 2.0),
               "offset": ("uniform", 0.0,2.0),
               "mean": ("uniform", -1.0,1.0),
               "std": ("uniform", 0.0,2.0)}


# In[5]:


_ = test_model_1.set_priors(priors_dict) # set priors for amplitude
priors_1 = test_model_1.get_sbi_priors() # get the priors object
theta_1 = priors_1.sample((n_simulations,)) # sample the priors to get parameters
_ = test_model_1.set_input_data(x) # set input data to produce simulations
simulator_1 = test_model_1.get_sbi_simulator() # get simulator object
y_1 =  simulator_1(theta_1) # extract observations from simulator object


# In[6]:


nle_sbi= NLESBI() # instantiate SBI class
inference_object_1 = nle_sbi.create_inference_object(priors_1) # create inference object
likelihood_net_1, posterior_distribution_1 = nle_sbi.compute_distribution(inference_object_1, theta_1, y_1) # get likelihood network


# In[7]:


distribution_theta = nle_sbi.sample_distribution(posterior_distribution_1, theta_1, observations)


# In[26]:


pairplot(distribution_theta, labels = list(priors_dict.keys()))


# In[8]:


potential_fn, _ = likelihood_estimator_based_potential(likelihood_net_1, priors_1, y_1, enable_transform=False)


# In[10]:


def log_likelihood_nautilus(param_dict):
    """
    Likelihood function for Nautilus using SBI-trained likelihood estimator.

    The key insight is that SBI's likelihood estimator is p(x|theta), so we need:
    - x: the observed data (y_obs)  
    - theta: the parameters we're evaluating
    """
    # Extract parameters
    numpy_theta = np.array([param_dict[key] for key in priors_dict.keys()])

    # Create parameter tensor
    theta = torch.tensor(numpy_theta, dtype=torch.float32)

    if len(theta.shape) == 1:
        theta = theta.unsqueeze(0)

    return potential_fn(theta).detach().numpy().flatten()[0]


# In[ ]:


prior_nautilus = Prior()

for key, value in priors_dict.items():
    prior_nautilus.add_parameter(key, dist=(value[1], value[2]))


sampler = Sampler(prior_nautilus, log_likelihood_nautilus, n_live=1000)
sampler.run(verbose=True)


# In[19]:


# Get likelihood
log_z_likelihood = sampler.log_z
print(log_z_likelihood)


# In[ ]:




