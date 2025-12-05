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
def compartment_rhs(x, t, disease_params, current_vaccination_rate, current_hes, npi_in_place):
    S,V,E,I = x
    beta, mu, c, gamma, delta = disease_params
    effective_beta = beta*(npi_in_place*npi_modifier + (1-npi_in_place))

    dS_dt = mu*(1-S)-current_vaccination_rate*(1-current_hes)*S-effective_beta*(c*E+I)*S
    dV_dt = (1-current_hes)*current_vaccination_rate*S-mu*V
    dE_dt = effective_beta*(c*E+I)*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.array([dS_dt, dV_dt, dE_dt, dI_dt])

def compartment_rhs_multi_patch(x, t, 
                                disease_params, current_vaccination_rate, current_hes, npi_in_place,
                                num_patches):
    # break state up into epidemiologically relevant categories
    S=x[:num_patches]
    V=x[num_patches:2*num_patches]
    E=x[2*num_patches:3*num_patches]
    I=x[3*num_patches:]

    effective_beta = np.zeros((num_patches, num_patches))
    beta, mu, c, gamma, delta = disease_params

    # modify disease spread if either patch has npi in place. Does not stack.
    for i in range(0, num_patches):
        for j in range(0, num_patches):
            either_npi = npi_in_place[i] or npi_in_place[j]
            effective_beta[i,j] = beta[i,j]*(either_npi*npi_modifier + (1-either_npi))

    dS_dt = mu*(1-S)-current_vaccination_rate*(1-current_hes)*S-np.matmul(effective_beta,(c*E+I))*S
    dV_dt = (1-current_hes)*current_vaccination_rate*S-mu*V
    dE_dt = np.matmul(effective_beta,(c*E+I))*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.concatenate((dS_dt, dV_dt, dE_dt, dI_dt))

# this method simulates one day of disease spread, given an initial state and
# an agent's choices for vaccination rate and beta (modified from a base level), as well as other
# parameters specific to the disease 
def simulate_day(initial_state, disease_params, current_vaccination_rate, current_hes, npi_in_place):
    return odeint(compartment_rhs_multi_patch, initial_state, [0, 1], args=(disease_params, current_vaccination_rate, current_hes, npi_in_place, num_patches))[-1,:]

def take_measurements(day, testing_schedule, decision_inputs, population_state, wastewater_used):
    # calculates how much is spent due to testing schedule
    costs_accrued = 0

    for in_patch in range(0, num_patches):
        # increment counters for measurements. If measurements are taken, they will be reset to zero anyway
        decision_inputs["time_since_diag"][in_patch] += 1/max_simulation_depth
        decision_inputs["time_since_wes"][in_patch] += 1/max_simulation_depth
        
        # extract testing schedule for the current patch
        diag_period = testing_schedule["diag_period"][in_patch]
        wes_period = testing_schedule["wes_period"][in_patch]

        if day % diag_period == 0:
            costs_accrued += cost_per_diag_measurement[in_patch]
            decision_inputs["time_since_diag"][in_patch] = 0
            decision_inputs["current_diag"][in_patch] = infected_seeking_care_frac[in_patch]*population_state[in_patch+3*num_patches]

        if day % wes_period == 0:
            costs_accrued += cost_per_wes_measurement[in_patch]
            decision_inputs["time_since_wes"][in_patch] = 0
            # if wastewater treatment plant has not been used to sample yet, add additional
            # 1-time cost
            if not wastewater_used[in_patch]:
                costs_accrued += cost_of_opening_wes_site[in_patch]
                wastewater_used[in_patch] = True
            
            # generate a noisy wastewater sample by exponentiating a lognormal sample (guarantees nonnegative)
            if population_state[in_patch+2*num_patches] > 0:
                measurement_mean = np.log(population_state[in_patch+2*num_patches]*max_pop[in_patch])
                decision_inputs["current_wes"][in_patch] = np.exp(rng.normal(measurement_mean, wes_std_frac[in_patch]*np.abs(measurement_mean)))/max_pop[in_patch]
            else:
                decision_inputs["current_wes"][in_patch] = 0

    # enforce upper limits correctly after incrementing counter
    decision_inputs["time_since_wes"][in_patch] = min(1, decision_inputs["time_since_wes"][in_patch])
    decision_inputs["time_since_diag"][in_patch] = min(1, decision_inputs["time_since_diag"][in_patch])
    
    return costs_accrued   

def affect_simulation(proposed_action, proposed_value, in_patch,
                      current_vaccination_rate, npi_in_place, sia_totals, population_state):
    
    # use selected action to modify simulation. Need to add check to ensure weird hacks don't emerge        
    if (proposed_action == "use_sia") and (sia_totals[in_patch] < sia_allowance[in_patch]):
        newly_vaxxed = population_state[in_patch]*sia_vax_fraction[in_patch]
        population_state[in_patch] -= newly_vaxxed
        population_state[num_patches+in_patch] += newly_vaxxed
        sia_totals[in_patch] += 1  # update total number of immunization actions used
    
    # elif proposed_action == "set_vax_rate":
    #     current_vaccination_rate[in_patch] = proposed_value*max_vax_rate[in_patch]
    
    # elif proposed_action == "apply_npi":
    #     npi_in_place[in_patch] = True
    
    # elif proposed_action == "remove_npi":
    #     npi_in_place[in_patch] = False

def generate_starting_day(method="geometric"):
    if method == "geometric":
        day_frac = rng.geometric(p=outbreak_prob)/max_simulation_depth
    elif method == "beta":
        day_frac = rng.beta(*outbreak_beta)
    
    return day_frac

def score_tree(candidate_tree, testing_schedule=default_schedule):
    # initialize a simulation
    outbreak_has_begun = False
    outbreak_has_ended = False
    wastewater_used = [False]*num_patches
    npi_in_place = [False]*num_patches
    current_vaccination_rate = starting_vax_rate
    sia_totals = [0]*num_patches
    population_state = np.concatenate((np.ones(num_patches), np.zeros(3*num_patches)))
    decision_inputs = dict(zip(simulation_outputs, starting_values))

    # results to return, keeps running totals on costs accrued by tree
    tree_score = 0

    # keeps tracks of all decision paths chosen by the tree, useful for data visualization
    decision_paths = []

    # keeps track of all actions, useful for data visualizations
    event_stream = []

    # determine day the outbreak will start
    starting_day_as_fraction = generate_starting_day(method=outbreak_method)
    
    for day in range(0, max_simulation_depth):
        if not outbreak_has_begun and (starting_day_as_fraction <= day/max_simulation_depth):
            outbreak_has_begun = True
            initial_exposed_pop = np.random.choice(range(1, 1+maximal_initial_exposed))
            initial_patch = np.random.choice(range(0, num_patches))
            # update susceptible and exposed category of relevant patch
            population_state[initial_patch] -= initial_exposed_pop/max_pop[initial_patch]
            population_state[initial_patch + num_patches*2] += initial_exposed_pop/max_pop[initial_patch]

        # use decision tree to generate a candidate action for the simulation
        proposed_action, proposed_value, in_patch, decision_path = candidate_tree.evaluate(decision_inputs)
        decision_paths.append(decision_path)
        event_stream.append([proposed_action, proposed_value, in_patch])
        
        # summary function used to decide how the simulation outputs and planner policy changes
        # relevant variables
        affect_simulation(proposed_action, proposed_value, in_patch, current_vaccination_rate, npi_in_place, sia_totals, population_state)
        
        # take measurements, to be used on the next day
        tree_score += take_measurements(day, testing_schedule, decision_inputs, population_state, wastewater_used)

        # compute cost of vaccine and npi interventions
        # current_hes = vax_hes_level(decision_inputs)
        current_hes = np.zeros(num_patches)
        
        # hard check to ensure no numerical leaking, even though none sick is a fixed point
        if outbreak_has_begun and not outbreak_has_ended:
            population_state = simulate_day(population_state, disease_params, current_vaccination_rate, current_hes, npi_in_place)
            # implement die-off of disease
            for in_patch in range(0, num_patches):
                perc_exposed = population_state[in_patch+2*num_patches]
                perc_infected = population_state[in_patch+3*num_patches]
                if (perc_exposed + perc_infected) < 1/max_pop[in_patch]:
                    population_state[in_patch] += perc_exposed + perc_infected
                    population_state[in_patch+2*num_patches] = 0
                    population_state[in_patch+3*num_patches] = 0
                    outbreak_has_ended = True

        # add costs of running totals
        for i in range(0, num_patches):
            tree_score += cost_per_exposed[i]*population_state[2*num_patches+i]*max_pop[i]
            tree_score += cost_per_infected[i]*population_state[3*num_patches+i]*max_pop[i]
            tree_score += cost_per_vax[i]*current_vaccination_rate[i]
            tree_score += cost_of_npi[i]*npi_in_place[i]
    
    return tree_score, decision_paths, event_stream

def tree_decision_plots(Tree, min_day=0, max_day=max_simulation_depth):
    sample_score, decision_path_samples, event_stream = score_tree(Tree)
    selected_samples = decision_path_samples[min_day:max_day]
    selected_events = event_stream[min_day:max_day]
    path_frequencies = dict(Counter(selected_samples).most_common())

    print(f"Sample score: {sample_score}\n")

    for key in path_frequencies.keys():
        if path_frequencies[key] > 0:
            print(f"{key} : {path_frequencies[key]}")

    current_vax = starting_vax_rate
    vax_rates = [[] for _ in range(0, num_patches)]
    
    for event in selected_events:
        if event[0] == "set_vax_rate":
            current_vax[event[2]] = event[1]
        for i in range(0, num_patches):
            vax_rates[i].append(current_vax[i])
    
    vax_rates = np.array(vax_rates)  # so we can slice this appropriately
    
    legend_labels = []
    for i in range(0, num_patches):
        plt.plot(vax_rates[i])
        legend_labels.append(f"Patch {i}")
    plt.title("Change in vaccination rates over sample run")
    plt.legend(legend_labels)
    plt.show()

def tree_sia_decisions(Tree):
    sample_score, decision_path_samples, event_stream = score_tree(Tree)

    sia_used = [0]*num_patches

    print(f"Sample score: {sample_score}\n")

    for i,event in enumerate(event_stream):
        if event[0] == "use_sia" and sia_used[event[2]] < sia_allowance[event[2]]:
            print(f"SIA implemented on day {i} in patch {event[2]}")
            print(f"Decision logic: {decision_path_samples[i]}")
            sia_used[event[2]] += 1