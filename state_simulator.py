import numpy as np
from trees import Node, DecisionTree, grow_random_tree
from parameters import *
from scipy.integrate import odeint
from collections import Counter
import matplotlib.pyplot as plt

# to generate random numbers
rng = np.random.default_rng()

# the RHS of our compartmental model for disease spread, used in simulate_day
# we exclude the R category via conservation laws to save compute
def compartment_rhs(x, t, disease_params, current_vaccination_rate, npi_in_place):
    S,V,E,I = x
    beta, mu, c, gamma, delta = disease_params
    effective_beta = beta*(npi_in_place*npi_modifier + (1-npi_in_place))

    dS_dt = mu*(1-S)-current_vaccination_rate*S-effective_beta*(c*E+I)*S
    dV_dt = current_vaccination_rate*S-mu*V
    dE_dt = effective_beta*(c*E+I)*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.array([dS_dt, dV_dt, dE_dt, dI_dt])

# this method simulates one day of disease spread, given an initial state and
# an agent's choices for vaccination rate and beta (modified from a base level), as well as other
# parameters specific to the disease 
def simulate_day(initial_state, disease_params, current_vaccination_rate, npi_in_place):
    return odeint(compartment_rhs, initial_state, [0, 1], args=(disease_params, current_vaccination_rate, npi_in_place))[-1,:]

def score_tree(candidate_tree, print_check=False):
    if print_check:
        print(f"{rng.uniform()}\n")
    
    # initialize a simulation
    outbreak_has_begun = False
    wastewater_used = False
    npi_in_place = False
    current_vaccination_rate = starting_vax_rate
    population_state = np.array([1, 0, 0, 0])
    decision_inputs = {"time_since_diag": 1,
                    "time_since_wes": 1,
                      "current_diag": 0,
                        "current_wes": 0}

    # results to return, keeps running totals on costs accrued by tree
    tree_score = 0

    # keeps tracks of all decision paths chosen by the tree, useful for data visualization
    decision_paths = []
    
    for day in range(0, max_simulation_depth):
        if not outbreak_has_begun and (rng.uniform() < outbreak_prob):
            outbreak_has_begun = True
            initial_exposed_pop = np.random.choice([x for x in range(1, 1+maximal_initial_exposed)])
            population_state = np.array([(max_pop-initial_exposed_pop)/max_pop, 0, initial_exposed_pop/max_pop, 0])

        # use decision tree to generate a candidate action for the simulation
        proposed_action, proposed_value, decision_path = candidate_tree.evaluate(decision_inputs)
        decision_paths.append(decision_path)
        
        # use selected action to modify simulation. Need to add check to ensure weird hacks don't emerge
        if proposed_action == "pass":
            # null action, simply update counters since measurement occurred
            decision_inputs["time_since_diag"] += 1/max_simulation_depth
            decision_inputs["time_since_wes"] += 1/max_simulation_depth
        
        elif proposed_action == "set_vax_rate":
            decision_inputs["time_since_diag"] += 1/max_simulation_depth
            decision_inputs["time_since_wes"] += 1/max_simulation_depth
            current_vaccination_rate = proposed_value*max_vax_rate
        
        elif proposed_action == "apply_npi":
            decision_inputs["time_since_diag"] += 1/max_simulation_depth
            decision_inputs["time_since_wes"] += 1/max_simulation_depth
            npi_in_place = True
        
        elif proposed_action == "remove_npi":
            decision_inputs["time_since_diag"] += 1/max_simulation_depth
            decision_inputs["time_since_wes"] += 1/max_simulation_depth
            npi_in_place = False
        
        elif proposed_action == "diagnostic_measurement":
            tree_score += cost_per_diag_measurement
            decision_inputs["time_since_diag"] = 0
            decision_inputs["current_diag"] = infected_seeking_care_frac*population_state[3]
        
        elif proposed_action == "wes_measurement":
            tree_score += cost_per_wes_measurement
            # if wastewater treatment plant has not been used to sample yet, add additional
            # 1-time cost
            if not wastewater_used:
                tree_score += cost_of_opening_wes_site
                wastewater_used = True
            
            # reset counter on time since measurement, and generate a noisy wastewater sample
            # by exponentiating a lognormal sample (guarantees nonnegative)
            decision_inputs["time_since_wes"] = 0
            if population_state[2] > 0:
                measurement_mean = np.log(population_state[2]*max_pop)
                decision_inputs["current_wes"] = np.exp(rng.normal(measurement_mean,wes_std_frac*np.abs(measurement_mean)))/max_pop
            else:
                decision_inputs["current_wes"] = 0
        
        # enforce upper limits correctly
        decision_inputs["current_wes"] = min(1, decision_inputs["current_wes"])
        decision_inputs["current_diag"] = min(1, decision_inputs["current_diag"])
        
        # hard check to ensure no numerical leaking, even though none sick is a fixed point
        if outbreak_has_begun:
            population_state = simulate_day(population_state, disease_params, current_vaccination_rate, npi_in_place)
        
        # compute cost of vaccine and npi interventions
        cost_per_vax = cost_vax_level(decision_inputs["current_diag"],
                                               decision_inputs["current_wes"],
                                               decision_inputs["time_since_diag"],
                                               decision_inputs["time_since_wes"])
        
        cost_of_npi = cost_npi_level(decision_inputs["current_diag"],
                                               decision_inputs["current_wes"],
                                               decision_inputs["time_since_diag"],
                                               decision_inputs["time_since_wes"])
        
        # add running totals
        tree_score += (cost_per_exposed*population_state[2]+cost_per_infected*population_state[3])*max_pop
        tree_score += cost_per_vax*current_vaccination_rate+cost_of_npi*npi_in_place
    
    return tree_score, decision_paths

def tree_decision_plots(Tree, min_day=0, max_day=max_simulation_depth):
    sample_score, decision_path_samples = score_tree(Tree)
    selected_samples = decision_path_samples[min_day:max_day]
    path_frequencies = dict(Counter(selected_samples).most_common())

    print(f"Sample score: {sample_score}\n")

    for key in path_frequencies.keys():
        if path_frequencies[key] > 0:
            print(f"{key} : {path_frequencies[key]}")

    plt.bar(range(len(path_frequencies)), list(path_frequencies.values()), align='center')
    plt.xticks(range(len(path_frequencies)), range(len(path_frequencies)))
    plt.show()
