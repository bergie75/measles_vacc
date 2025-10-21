import numpy as np

# other simulation details
max_simulation_depth = 180
max_pop = 5000

# possible actions and thresholds for decision trees
action_set = ["increase_vax_rate", "decrease_vax_rate", "apply_npi", "remove_npi", "diagnostic_measurement", "wes_measurement"]
input_threshold_ranges = {"time_since_diag": [0, max_simulation_depth],
                    "time_since_wes": [0, max_simulation_depth],
                      "current_diag": [0, max_pop],
                        "current_wes": [0, max_pop]}

# describes the overall range of choices on vaccination
vax_rate_modifier = 1.05

# how to step the npi measures
npi_modifier = 0.9

# affect measurements
wes_std_frac = 0.1  # determines how noisy wastewater surveillance is
infected_seeking_care_frac = 0.2

# disease dynamics parameters
mu = 1
c = 1
gamma = 1
delta = 1
beta = 0.1
vax_rate = 1
disease_params = np.array([beta,vax_rate,mu,c,gamma,delta])

# set outbreak initial conditions
outbreak_prob=0.01
maximal_initial_exposed=1/max_pop

# costs of various actions
cost_per_vax_increase = 1
cost_per_npi_increase = 1
cost_per_diag_measurement = 1
cost_per_wes_measurement = 1
cost_per_infected = 1
cost_per_exposed = 1

# parameters to control the genetic algorithm
num_simulations = 1000
max_tree_depth = 6
number_of_members = 1000
top_choices = 50
max_rounds = 100
threshold_mutation_probability = 0.001
decision_mutation_probability = 0.001