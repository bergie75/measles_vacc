import numpy as np

# other simulation details
max_simulation_depth = 720
max_pop = 5000

# possible actions and thresholds for decision trees
action_set = ["set_vax_rate",
               "apply_npi", "remove_npi",
               "diagnostic_measurement", "wes_measurement",
               "pass"]

# these actions will generate a proposed value, which must be set
actions_requiring_values = ["set_vax_rate"]

# describes the variables the decision tree can use
simulation_outputs = ["time_since_diag", "time_since_wes", "current_diag", "current_wes"]

# affect measurements
wes_std_frac = 0.001  # determines how noisy wastewater surveillance is
infected_seeking_care_frac = 0.2  # what fraction of patients who are infected seek medical attention

# disease dynamics parameters
mu = 0.01
c = 0.6
gamma = np.log(2)/3  # after 3 days, 50% of exposed patients become infected (half life)
delta = np.log(2)/3  # after 3 days, 50% of infected patients have recovered
beta = 0.3
starting_vax_rate = 0.001
disease_params = np.array([beta,mu,c,gamma,delta])

# describes the overall range of choices on vaccination
max_vax_rate = 1

# how to step the npi measures
npi_modifier = 0.5

# set outbreak initial conditions
outbreak_prob=1/150
maximal_initial_exposed=1

# vax hesitancy parameters
response_to_diag = 100
response_to_wes = 100
diag_info_decay_rate = -np.log(2)/4  # how long before the case count is considered half as impactful
wes_info_decay_rate = -np.log(2)/4  # how long before the case count is considered half as impactful
max_hes_frac = 1  # what fraction of the population could become hesitant

# cost functions for more complicated interventions
def vax_hes_level(current_diag, current_wes, time_since_diag, time_since_wes):
    wes_impact = response_to_wes*current_wes*np.exp(wes_info_decay_rate*time_since_wes*max_simulation_depth)
    diag_impact = response_to_diag*current_diag*np.exp(diag_info_decay_rate*time_since_diag*max_simulation_depth)
    # level of vaccine hesitancy
    hes_level = max_hes_frac*(1-np.tanh((wes_impact+diag_impact)*max_pop))  #argument greater than or equal to zero

    return hes_level

# costs of various actions
cost_per_vax = 100
cost_per_diag_measurement = 1
cost_per_wes_measurement = 0.5
cost_per_infected = 1000
cost_per_exposed = 1000
cost_of_npi = 1000
cost_of_opening_wes_site = 0

# parameters to control the genetic algorithm
num_simulations = 75
max_tree_depth = 4
number_of_members = 250
top_choices = 25
max_rounds = 200

# controls mutations in trees
threshold_mutation_probability = 0.9
threshold_attenuation = 0.98

action_mutation_probability = 0.9
action_attenuation = 0.98

decision_mutation_probability = 0.2
decision_attenuation = 0.98

variable_change_probability = 0.15
var_change_attenuation = 0.98

chop_decision_probability = 0.01
chop_attenuation = 1.01
