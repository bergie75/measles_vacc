import numpy as np

# other simulation details
max_simulation_depth = 720
max_pop = [5000]
num_patches = 1

# possible actions and thresholds for decision trees
action_set = ["set_vax_rate", "apply_npi", "remove_npi",]

# these actions will generate a proposed value, which must be set
actions_requiring_values = ["set_vax_rate"]

# describes the variables the decision tree can use
simulation_outputs = ["time_since_diag", "time_since_wes", "current_diag", "current_wes"]
starting_values = [[max_simulation_depth]*num_patches, [max_simulation_depth]*num_patches, np.zeros(num_patches), np.zeros(num_patches)]

# default measurement schedule, test every day in each patch
default_schedule = {}
default_schedule["wes_period"] = [1]*num_patches
default_schedule["diag_period"] = [1]*num_patches

# affect measurements
wes_std_frac = [0.001]  # determines how noisy wastewater surveillance is
infected_seeking_care_frac = [0.2]  # what fraction of patients who are infected seek medical attention

# disease dynamics parameters
mu = np.array([0.01]*num_patches)
c = np.array([0.6]*num_patches)
gamma = np.array([np.log(2)/3]*num_patches)  # after 3 days, 50% of exposed patients become infected (half life)
delta = np.array([np.log(2)/3]*num_patches)  # after 3 days, 50% of infected patients have recovered
beta = np.array([[0.3]])
starting_vax_rate = np.array([0.001]*num_patches)
disease_params = [beta,mu,c,gamma,delta]

# describes the overall range of choices on vaccination
max_vax_rate = [1]*num_patches

# how to step the npi measures
npi_modifier = 0.5

# set outbreak initial conditions
outbreak_prob=1/150
maximal_initial_exposed=1

# vax hesitancy parameters
response_to_diag = 0.1*np.eye(num_patches)
response_to_wes = 0.1*np.eye(num_patches)
diag_info_decay_rate = np.array([-np.log(2)/4]*num_patches)  # how long before the case count is considered half as impactful
wes_info_decay_rate = np.array([-np.log(2)/4]*num_patches)  # how long before the case count is considered half as impactful
max_hes_frac = np.array([1]*num_patches)  # what fraction of the population could become hesitant
min_hes_frac = np.array([0]*num_patches)  # some portion of population is always hesitant

# cost functions for more complicated interventions
def vax_hes_level(decision_inputs):
    # unpack values for convenience/brevity
    current_wes = decision_inputs["current_wes"]
    time_since_wes = decision_inputs["time_since_wes"]
    current_diag = decision_inputs["current_diag"]
    time_since_diag = decision_inputs["time_since_diag"]

    wes_impact = np.matmul(response_to_wes, current_wes)*np.exp(wes_info_decay_rate*time_since_wes*max_simulation_depth)
    diag_impact = np.matmul(response_to_diag, current_diag)*np.exp(diag_info_decay_rate*time_since_diag*max_simulation_depth)
    # level of vaccine hesitancy
    t = np.tanh((wes_impact+diag_impact)*max_pop)
    hes_level = min_hes_frac*t + max_hes_frac*(1-t)  #argument greater than or equal to zero

    return hes_level

# costs of various actions
cost_per_vax = [100]*num_patches
cost_per_diag_measurement = [1]*num_patches
cost_per_wes_measurement = [0.5]*num_patches
cost_per_infected = [1]*num_patches
cost_per_exposed = [0]*num_patches
cost_of_npi = [1000]*num_patches
cost_of_opening_wes_site = [0]*num_patches

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

patch_mutation_probability = 0.2
patch_attenuation = 0.98
