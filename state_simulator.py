import numpy as np
from parameters import exported_parameters
from collections import Counter
import matplotlib.pyplot as plt

# to generate random numbers
rng = np.random.default_rng()

def stochastic_day_update(population_state, local_params=exported_parameters):
    # unpack
    num_patches = local_params["num_patches"]
    disease_params = local_params["disease_params"]
    alpha = local_params["alpha"]

    beta, mu, c, gamma, delta = disease_params

    # use more convenient names for the population state, will update the original b/c of cloning by reference
    S=population_state[:num_patches]
    V=population_state[num_patches:2*num_patches]
    E=population_state[2*num_patches:3*num_patches]
    I=population_state[3*num_patches:4*num_patches]
    R=population_state[4*num_patches:]

    # we proceed using the Gillespie method. Set time of day to zero
    day_clock = 0
    
    # event rates should be given per hour
    while day_clock < 24:
        total_pop = S+V+E+I+R  # should always be approximately max_pop
        exposure_rates = np.matmul(beta, c*E+I)*S
        birth_rates = mu*total_pop
        death_rates = mu*total_pop  # a death could occur in any of the different compartments, we will divvy them up afterwards
        exp_to_inf_rates = gamma*E
        recovery_rates = delta*I
        vax_rates = alpha*S

        # find out the rate new events occur
        all_rates = np.concatenate((exposure_rates, vax_rates, exp_to_inf_rates, recovery_rates, birth_rates, death_rates))
        total_event_rate = np.sum(all_rates)
        prob_of_events = all_rates/total_event_rate

        # find new event and update population
        time_until_new_event = rng.exponential(1.0/total_event_rate)  # numpy uses scale instead of rate parameter
        day_clock += time_until_new_event
        event_index = int(rng.choice(len(all_rates), p=prob_of_events))

        event_type = event_index // num_patches
        event_patch = event_index % num_patches

        # event type determines where in the list of all_rates our event comes from. See the definition above.
        if event_type == 0:
            # an infection has occurred
            S[event_patch] -= 1
            E[event_patch] += 1
        elif event_type == 1:
            # someone has gotten vaccinated
            S[event_patch] -= 1
            V[event_patch] += 1
        elif event_type == 2:
            # an exposure has become an infection
            E[event_patch] -= 1
            I[event_patch] += 1
        elif event_type == 3:
            # an infected individual has recovered
            I[event_patch] -= 1
            R[event_patch] += 1
        elif event_type == 4:
            # an individual was born
            S[event_patch] += 1
        elif event_type == 5:
            # an individual has died, must pick a compartment
            patch_state = np.array([S[event_patch], V[event_patch], E[event_patch], I[event_patch], R[event_patch]])
            compartment = rng.choice(5, p=patch_state/total_pop[event_patch])
            
            # based on compartment choice, apply death
            if compartment == 0:
                S[event_patch] -= 1
            elif compartment == 1:
                V[event_patch] -= 1
            elif compartment == 2:
                E[event_patch] -= 1
            elif compartment == 3:
                I[event_patch] -= 1
            elif compartment == 4:
                R[event_patch] -= 1

def take_measurements(day, testing_schedule, decision_inputs, population_state, wastewater_used, local_params=exported_parameters):
    # unpack from config
    num_patches = local_params["num_patches"]
    max_simulation_depth = local_params["max_simulation_depth"]
    cost_per_diag_measurement = local_params["cost_per_diag_measurement"]
    cost_per_wes_measurement = local_params["cost_per_wes_measurement"]
    infected_seeking_care_frac = local_params["infected_seeking_care_frac"]
    cost_of_opening_wes_site = local_params["cost_of_opening_wes_site"]
    max_pop = local_params["max_pop"]
    wes_std_frac = local_params["wes_std_frac"]

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
            decision_inputs["current_diag"][in_patch] = infected_seeking_care_frac[in_patch]*population_state[in_patch+3*num_patches]/max_pop[in_patch]

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
                measurement_mean = np.log(population_state[in_patch+2*num_patches])
                decision_inputs["current_wes"][in_patch] = np.exp(rng.normal(measurement_mean, wes_std_frac[in_patch]*np.abs(measurement_mean)))/max_pop[in_patch]
            else:
                decision_inputs["current_wes"][in_patch] = 0

    # enforce upper limits correctly after incrementing counter
    decision_inputs["time_since_wes"][in_patch] = min(1, decision_inputs["time_since_wes"][in_patch])
    decision_inputs["time_since_diag"][in_patch] = min(1, decision_inputs["time_since_diag"][in_patch])
    
    return costs_accrued   

def affect_simulation(proposed_action, in_patch, sia_totals, population_state, local_params=exported_parameters):
    # unpack
    sia_allowance = local_params["sia_allowance"]
    sia_vax_fraction = local_params["sia_vax_fraction"]
    num_patches = local_params["num_patches"]
    
    # use selected action to modify simulation. Need to add check to ensure weird hacks don't emerge        
    if (proposed_action == "use_sia") and (sia_totals[in_patch] < sia_allowance[in_patch]):
        newly_vaxxed = int(round(population_state[in_patch]*sia_vax_fraction[in_patch]))
        population_state[in_patch] -= newly_vaxxed
        population_state[num_patches+in_patch] += newly_vaxxed
        sia_totals[in_patch] += 1  # update total number of immunization actions used

def generate_starting_day(method="geometric", local_params=exported_parameters):
    # unpack config
    outbreak_prob = local_params["outbreak_prob"]
    max_simulation_depth = local_params["max_simulation_depth"]
    outbreak_beta = local_params["outbreak_beta"]

    if method == "geometric":
        day_frac = rng.geometric(p=outbreak_prob)/max_simulation_depth
    elif method == "beta":
        day_frac = rng.beta(*outbreak_beta)
    
    return day_frac

def score_tree(candidate_tree, testing_schedule, local_params=exported_parameters):
    # unpack config
    num_patches = local_params["num_patches"]
    max_simulation_depth = local_params["max_simulation_depth"]
    max_pop = local_params["max_pop"]
    alpha = local_params["alpha"]
    simulation_outputs = local_params["simulation_outputs"]
    starting_values = local_params["starting_values"]
    outbreak_method = local_params["outbreak_method"]
    minimal_initial_exposed = local_params["minimal_initial_exposed"]
    maximal_initial_exposed = local_params["maximal_initial_exposed"]
    disease_params = local_params["disease_params"]
    cost_per_exposed = local_params["cost_per_exposed"]
    cost_per_infected = local_params["cost_per_infected"]
    cost_per_vax = local_params["cost_per_vax"]
    cost_of_npi = local_params["cost_of_npi"]

    # need mu to initialize vaccinated numbers
    _, mu, _, _, _ = disease_params

    # initialize a simulation
    outbreak_has_begun = False
    outbreak_has_ended = False
    wastewater_used = [False]*num_patches
    npi_in_place = [False]*num_patches
    sia_totals = [0]*num_patches
    decision_inputs = dict(zip(simulation_outputs, starting_values))

    # construct initial population
    population_state = [int(round(max_pop[i])*mu[i]/(mu[i]+alpha[i])) for i in range(0, num_patches)]  # set susceptible to equilibrium for vaccination
    initially_vaxxed = [max_pop[i]-population_state[i] for i in range(0, num_patches)]
    population_state.extend(initially_vaxxed)
    population_state.extend([0]*3*num_patches)  # accounts E,I,R
    population_state = np.array(population_state)

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
            initial_exposed_pop = np.random.choice(range(minimal_initial_exposed, 1+maximal_initial_exposed))
            initial_patch = np.random.choice(range(0, num_patches))
            # update susceptible and exposed category of relevant patch
            population_state[initial_patch] -= initial_exposed_pop
            population_state[initial_patch + num_patches*2] += initial_exposed_pop

        # use decision tree to generate a candidate action for the simulation
        proposed_action, proposed_value, in_patch, decision_path = candidate_tree.evaluate(decision_inputs)
        decision_paths.append(decision_path)
        event_stream.append([proposed_action, proposed_value, in_patch])
        
        # summary function used to decide how the simulation outputs and planner policy changes
        # relevant variables
        affect_simulation(proposed_action, in_patch, sia_totals, population_state)
        
        # take measurements, to be used on the next day
        tree_score += take_measurements(day, testing_schedule, decision_inputs, population_state, wastewater_used)
        
        # hard check to ensure no numerical leaking, even though none sick is a fixed point
        if outbreak_has_begun and not outbreak_has_ended:
            # use Gillespie method to simulate a day of the outbreak
            stochastic_day_update(population_state)
            
            # if disease has died off, can stop simulating disease dynamics
            sick_categories = population_state[2*num_patches:4*num_patches]
            if np.sum(sick_categories) == 0:
                outbreak_has_ended

        # add costs of running totals
        for i in range(0, num_patches):
            tree_score += cost_per_exposed[i]*population_state[2*num_patches+i]
            tree_score += cost_per_infected[i]*population_state[3*num_patches+i]
            tree_score += cost_per_vax[i]*alpha[i]
            tree_score += cost_of_npi[i]*npi_in_place[i]
    
    return tree_score, decision_paths, event_stream

def tree_decision_plots(Tree, min_day=0, max_day=1, local_params=exported_parameters):
    num_patches = local_params["num_patches"]
    alpha = local_params["alpha"]

    sample_score, decision_path_samples, event_stream = score_tree(Tree)
    selected_samples = decision_path_samples[min_day:max_day]
    selected_events = event_stream[min_day:max_day]
    path_frequencies = dict(Counter(selected_samples).most_common())

    print(f"Sample score: {sample_score}\n")

    for key in path_frequencies.keys():
        if path_frequencies[key] > 0:
            print(f"{key} : {path_frequencies[key]}")

    current_vax = alpha
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

def tree_sia_decisions(Tree, local_params=exported_parameters):
    # unpack config
    num_patches = local_params["num_patches"]
    sia_allowance = local_params["sia_allowance"]

    sample_score, decision_path_samples, event_stream = score_tree(Tree)

    sia_used = [0]*num_patches

    print(f"Sample score: {sample_score}\n")

    for i,event in enumerate(event_stream):
        if event[0] == "use_sia" and sia_used[event[2]] < sia_allowance[event[2]]:
            print(f"SIA implemented on day {i} in patch {event[2]}")
            print(f"Decision logic: {decision_path_samples[i]}")
            sia_used[event[2]] += 1