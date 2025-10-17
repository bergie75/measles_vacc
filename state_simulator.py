import numpy as np
from trees import Node, DecisionTree
from parameters import *
from scipy.integrate import odeint

# to generate random numbers
rng = np.random.default_rng()

# the RHS of our compartmental model for disease spread, used in simulate_day
# we exclude the R category via conservation laws to save compute
def compartment_rhs(x, t, disease_params, vax_rate, beta):
    S,V,E,I = x
    mu, c, gamma, delta = disease_params

    dS_dt = mu*(1-S)-vax_rate*S-beta*(c*E+I)*S
    dV_dt = vax_rate*S-mu*V
    dE_dt = beta*(c*E+I)*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.array([dS_dt, dV_dt, dE_dt, dI_dt])

# this method simulates one day of disease spread, given an initial state and
# an agent's choices for vaccination rate and npi_multiplier, as well as other
# parameters specific to the disease 
def simulate_day(initial_state, disease_params, vax_rate, beta):
    return odeint(compartment_rhs, initial_state, [0, 1], args=(disease_params, vax_rate, beta))[-1,:]

def score_tree(candidate_tree, outbreak_prob=0.01, maximal_initial_exposed=1/max_pop):
    # initialize a simulation
    outbreak_has_begun = False
    population_state = np.array([max_pop, 0, 0, 0])
    modifier_counters = {"vax": 0, "npi": 0}
    decision_inputs = {"time_since_diag": np.inf,
                    "time_since_wes": np.inf,
                      "current_diag": 0,
                        "current_wes": 0}
    vax_rate = default_vax_rate
    
    for day in range(0, max_simulation_depth):
        if not outbreak_has_begun and (rng.uniform < outbreak_prob):
            outbreak_has_begun = True
            initial_exposed_frac = rng.uniform(high=maximal_initial_exposed)
            population_state = np.array([max_pop*(1-initial_exposed_frac), max_pop*initial_exposed_frac, 0, 0])

        # use decision tree to generate a candidate action for the simulation
        proposed_action = candidate_tree.evaluate(decision_inputs)
        
        # use selected action to modify simulation. Need to add check to ensure weird hacks don't emerge
        if proposed_action == "increase_vax_rate":
            vax_rate *= vax_rate_modifier
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
            modifier_counters["vax"] += 1
        
        elif proposed_action == "decrease_vax_rate":
            # check if this action is legal
            if modifier_counters["vax"] > 0:
                vax_rate /= vax_rate_modifier
                modifier_counters["vax"] -= 1
            
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
        
        elif proposed_action == "apply_npi":
            beta *= npi_modifier
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
            modifier_counters["npi"] += 1
        
        elif proposed_action == "remove_npi":
            # check if this action is legal
            if modifier_counters["npi"] > 0:
                beta /= npi_modifier
                modifier_counters["npi"] -= 1
            
            decision_inputs["time_since_diag"] += 1
            decision_inputs["time_since_wes"] += 1
        
        elif proposed_action == "diagnostic_measurement":
            decision_inputs["time_since_diag"] = 0
            decision_inputs["current_diag"] = infected_seeking_care_frac*population_state[3]
        
        elif proposed_action == "wes_measurement":
            # reset counter on time since measurement, and generate a noisy wastewater sample
            # by exponentiating a lognormal sample (guarantees nonnegative)
            decision_inputs["time_since_wes"] = 0
            measurement_mean = np.log(population_state[2])
            decision_inputs["current_wes"] = np.exp(rng.lognormal(measurement_mean,wes_std_frac*measurement_mean))

        # hard check to ensure no numerical leaking, even though none sick is a fixed point
        if outbreak_has_begun:
            population_state = simulate_day(population_state, disease_params, vax_rate, beta)
