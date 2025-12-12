import numpy as np

# other simulation details
max_simulation_depth = 1000
num_patches = 1
max_pop = [5000]*num_patches

# possible actions and thresholds for decision trees
action_set = ["use_sia", "pass"]

# these actions will generate a proposed value, which must be set
actions_requiring_values = []

# describes the variables the decision tree can use
simulation_outputs = ["time_since_diag", "time_since_wes", "current_diag", "current_wes"]
starting_values = [np.zeros(num_patches), np.zeros(num_patches), np.zeros(num_patches), np.zeros(num_patches)]

# default measurement schedule, test every day in each patch
default_schedule = {}
default_schedule["wes_period"] = [1]*num_patches
default_schedule["diag_period"] = [1]*num_patches

# set allowance for supplemental immunization activity
sia_per_patch = 1
sia_allowance = [sia_per_patch]*num_patches

# dynamics of an sia
sia_vax_fraction = [0.3]*num_patches

# affect measurements
wes_std_frac = [0.001]*num_patches  # determines how noisy wastewater surveillance is
infected_seeking_care_frac = [0.2]*num_patches  # what fraction of patients who are infected seek medical attention

# disease dynamics parameters, division through by 24 to put in terms of hours for stochastics
mu = np.array([0.001/24]*num_patches)
c = np.array([0]*num_patches)
gamma = np.array([np.log(2)/72]*num_patches)  # 3 day average asymptomatic half life
delta = np.array([np.log(2)/72]*num_patches)  # 3 day average recovery half life
beta = np.array([[0.0014/24]])
alpha = np.array([0.012/24]*num_patches)  # vaccination parameter
waning_rate = np.array([0.003/24]*num_patches)
disease_params = [beta,mu,c,gamma,delta]

# describes the overall range of choices on vaccination
max_vax_rate = [1]*num_patches

# how to step the npi measures
npi_modifier = 0.5

# choose the method to start the outbreak and how many are exposed
outbreak_method = "geometric"
minimal_initial_exposed=1
maximal_initial_exposed=10

# set outbreak initial conditions
outbreak_prob=1/100 # for geometric distribution
outbreak_beta=[7,3] # bias arrivals towards a certain point in sims with beta distribution

# costs of various actions
cost_per_vax = [0]*num_patches
cost_per_diag_measurement = [0]*num_patches
cost_per_wes_measurement = [0]*num_patches
cost_per_infected = [1]*num_patches
cost_per_exposed = [0]*num_patches
cost_of_npi = [0]*num_patches
cost_of_opening_wes_site = [0]*num_patches

# parameters to control the genetic algorithm
num_simulations = 300
max_tree_depth = 4
number_of_members = 60
top_choices = 15
max_rounds = 250

# controls bias in tree formation towards lower threshold values
use_beta_generation = False
dist_alpha = 1
dist_beta = 19

# controls mutations in trees
max_mutate_perc = 1

threshold_mutation_probability = 0.99
threshold_attenuation = 0.98

action_mutation_probability = 0.2
action_attenuation = 0.98

decision_mutation_probability = 0.2
decision_attenuation = 0.98

variable_change_probability = 0.15
var_change_attenuation = 0.98

chop_decision_probability = 0.01
chop_attenuation = 1.01

patch_mutation_probability = 0.0
patch_attenuation = 0.98

# packages all variables to be delivered to other files in a changeable way
variable_list = ["max_simulation_depth", "num_patches", "max_pop",
                 "action_set", "actions_requiring_values", "simulation_outputs",
                 "starting_values", "default_schedule", "sia_per_patch",
                 "sia_allowance", "sia_vax_fraction", "wes_std_frac",
                 "infected_seeking_care_frac", "alpha", "disease_params",
                 "max_vax_rate", "npi_modifier", "outbreak_method",
                 "minimal_initial_exposed", "maximal_initial_exposed", "outbreak_prob", "outbreak_beta",
                 "cost_per_vax", "cost_per_diag_measurement", "cost_per_wes_measurement",
                 "cost_per_infected", "cost_per_exposed", "cost_of_npi",
                 "cost_of_opening_wes_site", "num_simulations", "max_tree_depth",
                 "number_of_members", "top_choices", "max_rounds",
                 "use_beta_generation", "dist_alpha", "dist_beta",
                 "max_mutate_perc", "threshold_mutation_probability",
                 "threshold_attenuation", "action_mutation_probability",
                 "action_attenuation", "decision_mutation_probability",
                 "decision_attenuation", "variable_change_probability",
                 "var_change_attenuation", "patch_mutation_probability",
                 "patch_attenuation", "chop_decision_probability", "chop_attenuation",
                 "waning_rate"]

variable_vals = [max_simulation_depth, num_patches, max_pop,
                 action_set, actions_requiring_values, simulation_outputs,
                 starting_values, default_schedule, sia_per_patch,
                 sia_allowance, sia_vax_fraction, wes_std_frac,
                 infected_seeking_care_frac, alpha, disease_params,
                 max_vax_rate, npi_modifier, outbreak_method,
                 minimal_initial_exposed, maximal_initial_exposed, outbreak_prob, outbreak_beta,
                 cost_per_vax, cost_per_diag_measurement, cost_per_wes_measurement,
                 cost_per_infected, cost_per_exposed, cost_of_npi,
                 cost_of_opening_wes_site, num_simulations, max_tree_depth,
                 number_of_members, top_choices, max_rounds,
                 use_beta_generation, dist_alpha, dist_beta,
                 max_mutate_perc, threshold_mutation_probability,
                 threshold_attenuation, action_mutation_probability,
                 action_attenuation, decision_mutation_probability,
                 decision_attenuation, variable_change_probability,
                 var_change_attenuation, patch_mutation_probability,
                 patch_attenuation, chop_decision_probability, chop_attenuation,
                 waning_rate]

# all files will call this and unpack it locally, so changes can be made easily and consistently
# elsewhere
exported_parameters = dict(zip(variable_list, variable_vals))