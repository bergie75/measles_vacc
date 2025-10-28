import numpy as np
from trees import Node, DecisionTree, grow_random_tree
from parameters import *
from scipy.integrate import odeint

# to generate random numbers
rng = np.random.default_rng()

# the RHS of our compartmental model for disease spread, used in simulate_day
# we exclude the R category via conservation laws to save compute
def compartment_rhs(x, t, disease_params, vax_rate_modifier, npi_modifier, modifier_counters):
    S,V,E,I = x
    beta, vax_rate, mu, c, gamma, delta = disease_params
    effective_beta = beta*(npi_modifier**modifier_counters["npi"])
    effective_vax_rate = vax_rate*(vax_rate_modifier**modifier_counters["vax"])

    dS_dt = mu*(1-S)-effective_vax_rate*S-effective_beta*(c*E+I)*S
    dV_dt = effective_vax_rate*S-mu*V
    dE_dt = effective_beta*(c*E+I)*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.array([dS_dt, dV_dt, dE_dt, dI_dt])

# this method simulates one day of disease spread, given an initial state and
# an agent's choices for vaccination rate and beta (modified from a base level), as well as other
# parameters specific to the disease 
def simulate_day(initial_state, disease_params, vax_rate_modifier, npi_modifier, modifier_counters):
    return odeint(compartment_rhs, initial_state, [0, 1], args=(disease_params, vax_rate_modifier, npi_modifier, modifier_counters))[-1,:]

def score_tree(candidate_tree):
    # initialize a simulation
    outbreak_has_begun = False
    wastewater_used = False
    population_state = np.array([max_pop, 0, 0, 0])
    modifier_counters = {"vax": 0, "npi": 0}
    decision_inputs = {"time_since_diag": np.inf,
                    "time_since_wes": np.inf,
                      "current_diag": 0,
                        "current_wes": 0}

    # results to return, keeps running totals on costs accrued by tree
    tree_score = 0
    
    for day in range(0, max_simulation_depth):
        if not outbreak_has_begun and (rng.uniform() < outbreak_prob):
            outbreak_has_begun = True
            initial_exposed_pop = np.random.choice([x for x in range(1, 1+maximal_initial_exposed)])
            population_state = np.array([max_pop-initial_exposed_pop, 0, initial_exposed_pop, 0])

        # use decision tree to generate a candidate action for the simulation
        proposed_action = candidate_tree.evaluate(decision_inputs)
        
        # use selected action to modify simulation. Need to add check to ensure weird hacks don't emerge
        if proposed_action == "increase_vax_rate":
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
            modifier_counters["vax"] += 1
        
        elif proposed_action == "decrease_vax_rate":
            # check if this action is legal
            if modifier_counters["vax"] > 0:
                modifier_counters["vax"] -= 1
            
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
        
        elif proposed_action == "apply_npi":
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
            modifier_counters["npi"] += 1
        
        elif proposed_action == "remove_npi":
            # check if this action is legal
            if modifier_counters["npi"] > 0:
                modifier_counters["npi"] -= 1
            
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
        
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
                measurement_mean = np.log(population_state[2])
                decision_inputs["current_wes"] = np.exp(rng.lognormal(measurement_mean,wes_std_frac*np.abs(measurement_mean)))
            else:
                decision_inputs["current_wes"] = 0
        
        # hard check to ensure no numerical leaking, even though none sick is a fixed point
        if outbreak_has_begun:
            population_state = simulate_day(population_state, disease_params, vax_rate_modifier, npi_modifier, modifier_counters)
        
        # compute cost of vaccine interventions
        cost_per_vax_increase = cost_vax_level(decision_inputs["current_diag"],
                                               decision_inputs["current_wes"],
                                               decision_inputs["time_since_diag"],
                                               decision_inputs["time_since_wes"])
        
        # add running totals
        tree_score += cost_per_exposed*population_state[2]+cost_per_infected*population_state[3]
        tree_score += cost_per_vax_increase*modifier_counters["vax"]+cost_per_npi_increase*modifier_counters["npi"]
    
    return tree_score

if __name__ == "__main__":
    pass