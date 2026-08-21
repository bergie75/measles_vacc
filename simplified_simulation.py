# logistical/os/plotting packages
import time
import os
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

# mathematics packages
from random import sample
import random
import numpy as np
import pandas as pd
import math
from scipy.integrate import odeint

# define sources of randomness
random.seed(42) #not needed but added to compare with other code templates
rng = np.random.default_rng()

# loading data and pre-computed values for main process
print("[SETUP] Starting file loading and preprocessing...")
setup_start = time.perf_counter()

file_name = "kentucky-counties-by-population-(2026) all Ys.xlsx"
df = pd.read_excel(file_name)

# Normalize column names
df.columns = df.columns.str.strip()

#Rename columns for easier access
realworld_example = df.rename(columns={ 
    'county': 'Patch',
    'pop2026': 'Patch Size',
    'Survielled': 'Survielled',
    'Clustr': 'Cluster'
})
# Convert non-numeric cluster IDs to integers 0..N-1
cluster_codes, unique_clusters = pd.factorize(realworld_example['Cluster'])
realworld_example['Cluster'] = cluster_codes

# Operational surveillance boolean list based on realworld_example 'Survielled' column
realworld_operational_surveillance = [str(x).strip().upper() == 'Y' for x in realworld_example['Survielled']]
adjacent_matrix = pd.read_csv("adjacent weighted matrix.csv")
adjacent_matrix = adjacent_matrix.set_index("Unnamed: 0")
print(f"[SETUP] Task completed in {time.perf_counter() - setup_start:.4f} seconds.")

class Storm_Cell:
    def __init__(self, intensity, start_time, end_time, storm_origin):
        self.start_time = start_time
        self.end_time = end_time
        self.intensity = intensity
        self.storm_origin = storm_origin

def find_true(col):
    try:
        return list(col).index(True)
    except:
        return len(col)-1

# definition of the compartmental model used for simulations
# includes waning immunity and breakthrough infections
def compartment_rhs_multi_patch(x, t, disease_params, num_patches, N):
    
    # break state up into epidemiologically relevant categories
    #x is all subpopulations
    S1=x[0:num_patches]
    S2=x[num_patches:2*num_patches]
    V=x[2*num_patches:3*num_patches]
    E=x[3*num_patches:4*num_patches]
    I=x[4*num_patches:5*num_patches]
    cum_cases = x[5*num_patches:6*num_patches]

    # omega: rate of losing vaccine immunity (V -> S2)
    # sigma: relative susceptibility/breakthrough factor for vaccinated individuals
    alpha, beta, mu, c, gamma, delta, omega, sigma= disease_params
    
    # useful intermediate variable to improve efficiency
    force_of_infection = np.matmul(beta,(c*E+I)/N)

    dS1_dt = mu*(N-S1)-alpha*S1-force_of_infection*S1 #mu*(N-S) represents birth minus death rate
    dS2_dt = omega*V - force_of_infection*S2 - mu*S2    #beta is a big matrix that specifies how patches are interacting
    dV_dt = alpha * S1 - omega * V - sigma * force_of_infection * V - mu * V                    # with each other, mu*V are people dying while vaccinated            
    dE_dt = force_of_infection*(S1+S2+sigma*V)-mu*E- gamma*E
    dI_dt = gamma*E-(mu+delta)*I
    dcum_cases_dt = I #keep tracking of the cumulative case count
    
    return np.concatenate([dS1_dt, dS2_dt, dV_dt, dE_dt, dI_dt, dcum_cases_dt])

# define rainfall patterns over course of simulation
# uses Poisson cluster model, with a matrix to distribute the effects of rainfall
# greek letters come from the paper I am using as a guide, may not be standard
# rates should be in days
def rainfall_realization(storm_arrival_lambda, storm_duration_gamma, cell_arrival_beta, cell_duration_eta, rainfall_intensity,
                          t, geo_connectivity):
    
    print("  [RAINFALL] Starting rainfall realization generation...")
    func_start = time.perf_counter()
    
    num_patches = len(storm_arrival_lambda)  # each patch gets its own arrival rate
    num_timepoints = len(t)

    # fill with stochastic realizations
    rainfall = np.zeros((num_patches, num_timepoints))

    # rate parameter of storm event
    total_storm_lambda = np.sum(storm_arrival_lambda)
    storm_patch_probabilities = storm_arrival_lambda/total_storm_lambda

    # generate all storms, with lifetimes
    current_time = 0
    end_time = max(t)

    cell_list = []

    while current_time < end_time:
        dt = rng.exponential(1/total_storm_lambda)
        storm_origin_patch = np.random.choice(num_patches, p=storm_patch_probabilities)  # independent exponential random variables property
        current_time += dt
        
        initial_cell = Storm_Cell(rainfall_intensity[storm_origin_patch]*rng.uniform(),current_time,
                                  current_time+rng.exponential(1/cell_duration_eta[storm_origin_patch]),
                                  storm_origin_patch)
        cell_list.append(initial_cell)

        # determine number of storm cells, storm duration
        storm_start_time = current_time
        storm_end_time = current_time + rng.exponential(1/storm_duration_gamma[storm_origin_patch])
        cell_geom_prob = storm_duration_gamma[storm_origin_patch]/(storm_duration_gamma[storm_origin_patch]+cell_arrival_beta[storm_origin_patch])
        num_cells_to_generate = rng.geometric(cell_geom_prob)-1

        while num_cells_to_generate > 0:
            # see if this cell arrival is valid, i.e. it occurs before the end of the storm. Otherwise just try again
            trial_cell_start = rng.exponential(1/cell_arrival_beta[storm_origin_patch]) + storm_start_time
            if trial_cell_start < storm_end_time:
                num_cells_to_generate -= 1  # we accept this cell as valid
                trial_cell_end = trial_cell_start + rng.exponential(1/cell_duration_eta[storm_origin_patch])
                new_cell = Storm_Cell(rainfall_intensity[storm_origin_patch]*rng.uniform(), trial_cell_start,
                                      trial_cell_end, storm_origin_patch)
                cell_list.append(new_cell)

    # sort array based on arrival time of each cell instead of the storm it is part of
    cell_list.sort(key=lambda x: x.start_time)

    # Tima's modification to the rain realization code 
    for raincell in cell_list:
        
        start_idx = max(0, int(raincell.start_time))  
        end_idx = min(num_timepoints - 1, int(raincell.end_time))
        
        if start_idx==end_idx:

            rainfall[raincell.storm_origin,start_idx] += raincell.intensity*(raincell.end_time-raincell.start_time)
    
        else:
            
            rainfall[raincell.storm_origin,start_idx] += raincell.intensity*(math.ceil(raincell.start_time)-raincell.start_time)
            rainfall[raincell.storm_origin,end_idx] += raincell.intensity*(raincell.end_time-math.ceil(raincell.start_time))

    result = np.matmul(geo_connectivity, rainfall)
    print(f"  [RAINFALL] Task completed in {time.perf_counter() - func_start:.4f} seconds.")
    return np.matmul(geo_connectivity, rainfall)  # take geographic structure into account

def sensitivity_site_closure(df, sensitivities, closure_perc):
    func_start = time.perf_counter()
    num_patches = len(df)
    n_clusters = df["Cluster"].nunique()

    # Identify indices of counties flagged as 'Y' in Excel
    matching_rows = df[realworld_operational_surveillance].index
    
    operational_surveillance = [i in matching_rows for i in range(num_patches)]
    
    total_sensitivities = np.zeros(n_clusters)
    cluster_prob_dicts = [{} for _ in range(n_clusters)]
    
    # Accumulate sensitivities for active Excel sites in each cluster
    for i in matching_rows:
        cluster_id = df.loc[i, "Cluster"]
        total_sensitivities[cluster_id] += sensitivities[i]
    
    # Compute relative probabilities for closure selection within each cluster
    for i in matching_rows:
        cluster_id = df.loc[i, "Cluster"]
        if total_sensitivities[cluster_id] > 0:
            cluster_prob_dicts[cluster_id][i] = sensitivities[i] / total_sensitivities[cluster_id]
    
    # Randomly select active sites to close per cluster according to closure_perc
    for j in range(n_clusters):
        if len(cluster_prob_dicts[j]) > 0:
            num_stations_to_close = round(closure_perc * len(cluster_prob_dicts[j]))
            if num_stations_to_close > 0:
                stations_closed = np.random.choice(
                    list(cluster_prob_dicts[j].keys()), 
                    size=num_stations_to_close,
                    replace=False, 
                    p=list(cluster_prob_dicts[j].values())
                )
                for station in stations_closed:
                    operational_surveillance[station] = False
    
    return operational_surveillance

def calculate_patch_R0(patch_populations,disease_param, multi_beta):

    (multi_alpha,
        multi_beta,
        multi_mu,
        multi_c,
        multi_gamma,
        multi_delta,
        multi_omega,
        multi_sigma,
    ) = disease_param
    
    N = np.array(patch_populations)

    main_diag = np.diag(multi_beta)/patch_populations

    S1_star = (multi_mu * N) / (multi_mu + multi_alpha)

    S2_star = (multi_omega * multi_alpha * N) / ((multi_mu + multi_alpha) * (multi_mu + multi_omega))

    V_star = (multi_mu * multi_alpha * N) / ((multi_mu + multi_alpha) * (multi_mu + multi_omega))

    epi_factor = (main_diag * (multi_c * (multi_mu + multi_delta) + multi_gamma)) / ((multi_mu + multi_gamma) * (multi_mu + multi_delta))

    R0 = epi_factor * (S1_star + S2_star + multi_sigma * V_star)

    return R0

# maybe make rho_max 4.36*np.log(10)
def vaccination_strategy_better_wes_model(tag, abridged_disease_params, patch_populations, extra_doses_per_day, rainfall_matrix,
                                           sensitivities=0.03, num_days=1500, rho_max=1, V_p=1,
                                           chosen_patch=0, initial_cluster_allocation=1, operational_surveillance=None, detection_day_lag=0,
                                           days_to_disperse_stockpile=0, multi_beta=1):
    
    sim_start_time = time.perf_counter()
    print("    [SIMULATION] Starting vaccination_strategy_better_wes_model simulation...")
    
    # find files to load data
    script_dir = Path(__file__).resolve().parent
    save_folder = script_dir / "test 3 beta 0.00001, RI=0" / tag

    # combine preloaded beta with other disease parameters
    multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma = deepcopy(abridged_disease_params)
    disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]

    # important to ensure this is a numpy array
    patch_populations = np.array(patch_populations)
    kappa = -np.mean(multi_gamma)*np.log(1-0.5)

    # compute population in each cluster
    cluster_populations = np.zeros(n_clusters)
    patches_per_cluster = np.zeros(n_clusters)
    
    for j in range(0, num_patches):
        cluster_populations[clusters[j]] += patch_populations[j]
        patches_per_cluster[clusters[j]] += 1
    
    # expand test sensitivities if needed
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = sensitivities*np.ones(num_patches)
    
    # if no site closures specified, assume all WES sites are operational
    if operational_surveillance is None:
        operational_surveillance = [True]*num_patches

    # calculate basic reproduction number
    R_0=calculate_patch_R0(patch_populations, disease_params, multi_beta)
    
    # prepapre the initial state for the simulation
    pop_state = np.zeros(6*num_patches)
    pop_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
    pop_state[num_patches:2*num_patches] = ((multi_alpha*multi_omega)/((multi_mu+multi_alpha)*(multi_mu+multi_omega)))*patch_populations
    pop_state[2*num_patches:3*num_patches] = ((multi_alpha*multi_mu)/((multi_mu+multi_alpha)*(multi_mu+multi_omega)))*patch_populations
    frac_exposed = 0.01*pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[3*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state
    track_cumulative_totals = np.zeros(num_patches)
    
    # track which clusters have currently registered a detection event
    detections_in_cluster = [False]*n_clusters
    sia_intervention_allocated = False
    stockpile_days = None  # need to do this to avoid non-assignment area later down
    evident_infections_date = np.array([np.inf]*num_patches)

    # TRACKING VARIABLES FOR DETECTION
    first_detection_day = None
    first_detection_patch = None    

    # TRACKING ADDITIONAL ALPHA
    additional_alpha_tracked = np.zeros(num_patches)

    ode_daily_times = np.zeros(num_days)

    for day in range(0, num_days):
        # simulate one day of disease evolution
        if day % 100 == 0 or day == num_days - 1:
            print(f"      [SIMULATION] Processing Day {day}/{num_days}...")

        t_ode_start = time.perf_counter()
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1,:]
        ode_day_duration = time.perf_counter() - t_ode_start
    
        ode_daily_times[day] = ode_day_duration
    
        susceptible = pop_state[:num_patches]               # This is S1
        waned_susceptible = pop_state[num_patches:2*num_patches] # This is S2
        vaccinated = pop_state[2*num_patches:3*num_patches] # This is V
        exposures = pop_state[3*num_patches:4*num_patches]  # This is E
        infected = pop_state[4*num_patches:5*num_patches]

        for i,date in enumerate(evident_infections_date):
            if not np.isfinite(date) and infected[i]>=1: #we are checking if there are any cases of infection
                evident_infections_date[i] = day #if true, the date of that day is stored

        # we get one sia per simulation, check if it has been used
        if not sia_intervention_allocated:
            # generate wes data and check for detections
            wes_numerator = ((1-np.exp(-R_0*(np.maximum(day-evident_infections_date,0))))*(1-gamma/(gamma+kappa))*exposures+infected)*rho_max*V_p #total RNA in water, should be zero if no infection present
            wes_data = wes_numerator/(patch_populations*V_p+rainfall_matrix[:,day])
            # rho_max is the maximam amount that an exposed person could shed RNA into the water
            # detections modified to ensure site has been selected to continue functioning

            potential_detections = [(wes >= sensitivities[j]*rho_max) and operational_surveillance[j] for j,wes in enumerate(wes_data)]

            if any(potential_detections) and first_detection_day is None:
                # Find the index of the first patch that triggered the detection
                first_detection_patch = potential_detections.index(True)
                first_detection_day = day

            # update cluster detection tracking

            for j,detection_status in enumerate(potential_detections):
                detections_in_cluster[clusters[j]] = detections_in_cluster[clusters[j]] or detection_status

            # select all detection events to use as a potential initial location for outbreak
            # multiple can occur simultaneously
            cluster_indices_with_detections = []

            for j,detected_in_cluster in enumerate(detections_in_cluster):
                if detected_in_cluster:
                    cluster_indices_with_detections.append(j)

            # outbreak has been found, implement selected vaccination strategy
            if len(cluster_indices_with_detections) > 0:
                sia_intervention_allocated = True
                countdown_to_vaccination = detection_day_lag
                ground_zero_cluster = np.random.choice(cluster_indices_with_detections)  # most of the time this is picking from a list of length one?

                # this code is to determine how many vaccine doses we have stockpiled based on our initial closings, which we can use
                # "immediately"
                stockpiled_doses = day*extra_doses_per_day
                stockpiled_per_patch = stockpiled_doses/patches_per_cluster[ground_zero_cluster]
                if days_to_disperse_stockpile == 0:
                    stockpiled_per_patch_per_day = 0
                else:
                    stockpiled_per_patch_per_day = stockpiled_per_patch/days_to_disperse_stockpile
                
                # divide sia budget between patches, using allocation to decide how much goes to ground zero
                # vs other patches
                initial_cluster_budget = initial_cluster_allocation*extra_doses_per_day
                other_clusters_budget = (1-initial_cluster_allocation)*extra_doses_per_day

                # useful variables to divide resources within patches
                cluster_populations = np.zeros(n_clusters)

                for j in range(0, num_patches):
                    cluster_populations[clusters[j]] += patch_populations[j]

        # we have started the countdown until the detection is revealed and we act
        if sia_intervention_allocated:
            # this should only trigger once, implements delay from site sampling to detection announcement
            if countdown_to_vaccination == 0:
                # begin dispersing the emergency stockpile
                stockpile_days = days_to_disperse_stockpile
                # loop over all patches and provide allocated vaccination resources
                non_origin_total_pop = np.sum(cluster_populations)-cluster_populations[ground_zero_cluster]

                for j in range(0, num_patches):
                    if clusters[j] == ground_zero_cluster:
                        additional_alpha = initial_cluster_budget/cluster_populations[ground_zero_cluster]  # distribute according to population size
                        multi_alpha[j] += additional_alpha
                        additional_alpha_tracked[j] += additional_alpha

                    else:
                        additional_alpha = other_clusters_budget/non_origin_total_pop
                        multi_alpha[j] += additional_alpha
                        additional_alpha_tracked[j] += additional_alpha
                    

            # do this at the end so that of this if statement so that a lag of zero actually means zero
            countdown_to_vaccination -= 1
        
        # we get to use our built-up stockpile of vaccines over a very short number of days, unless we have disabled this mode
        # by setting the number of days to disperse our stockpile to zero
        if (stockpile_days is not None) and stockpile_days > 0:

            for j in range(0, num_patches):
                if clusters[j] == ground_zero_cluster:
                    susceptible[j] -= min(stockpiled_per_patch_per_day, susceptible[j])

    track_cumulative_totals = pop_state[5*num_patches:]
    
    total_time = time.perf_counter() - sim_start_time
    
    print(f"    [SIMULATION] Task completed in {total_time:.4f} seconds.") 
                    
    return track_cumulative_totals, clusters, first_detection_day, first_detection_patch, additional_alpha_tracked

def generating_beta_matrix(adj_matrix, raw_beta, in_cluster_strength):
   
    print("  [BETA MATRIX] Starting beta matrix generation...")
    func_start = time.perf_counter() 
   
    matrix_max = adj_matrix.max() if isinstance(adj_matrix, np.ndarray) else adj_matrix.values.max()
    normalized_matrix= (adj_matrix) / (matrix_max)

    matrix_modified = normalized_matrix*in_cluster_strength
    shape=matrix_modified.shape
    matrix = matrix_modified + np.eye(shape[0])
    
    #splitting the matrix depending on how many counties are divided into how many subpart
    
    labels = list(matrix.index)
    new_labels = []
    
    splits = {
        "Jefferson County": 5,
        "Fayette County": 2
    }
    
    for county in labels:
        if county in splits:
            for i in range(1, splits[county] + 1):
                new_labels.append(f"{county}_{i}")
        else:
            new_labels.append(county)

    matrix_split = pd.DataFrame(0.0, index=new_labels, columns=new_labels)
    
    for r_label in new_labels:
        for c_label in new_labels:
            # Get original parent county names
            orig_row = r_label.rsplit('_', 1)[0] if r_label.rsplit('_', 1)[0] in splits else r_label
            orig_col = c_label.rsplit('_', 1)[0] if c_label.rsplit('_', 1)[0] in splits else c_label
    
    #Same county or subparts of the same county -> full internal coupling (1.0)
            if orig_row == orig_col:
                matrix_split.loc[r_label, c_label] = 1.0
    
    #Different counties -> scale off-diagonal flow by product of subpart counts
            else:
                r_factor = splits.get(orig_row, 1)
                col_factor = splits.get(orig_col, 1)
    
    #Take off-diagonal value from  matrix
                val = matrix.loc[orig_row, orig_col]
                matrix_split.loc[r_label, c_label] = val / (r_factor * col_factor)
    
    beta_matrix=matrix_split*raw_beta
    
    print(f"  [BETA MATRIX] Task completed in {time.perf_counter() - func_start:.4f} seconds.")
    return(beta_matrix)

if __name__ == "__main__":
    tag = "debugging_improved_wes_model"
    run_through = True
    cluster_first = True

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    
    save_folder = base_dir / "test 3 beta 0.00001, RI=0" / tag
    save_folder.mkdir(parents=True, exist_ok=True)
    save_folder_str = str(save_folder)


    # patch connection strengths
    in_patch = 10**(-2)
    out_patch = in_patch*10**(-1)

    # cluster information
    num_patches = len(realworld_example)
    clusters = np.array(realworld_example["Cluster"])
    orig_n_clusters = len(np.unique(clusters))
    n_clusters = realworld_example["Cluster"].nunique()  # try extra clusters

    # other disease parameters, given on day timescale
    gamma = np.log(1/0.9)/7
    delta = np.log(1/0.9)/10
    mu = 0.01
    c = 0.2
    alpha = 0.1
    beta = 0.1/10000 #how likely is an infected person to spread the disease to someone else
    omega = 0.0001
    sigma=0.0001

    # for monitoring
    num_days = 600
    detection_threshold = 0.07
    site_to_dose_conversion = 7029
    detect_lag=4
    stockpile_dispersal_days = 0

    # calculate a rainfall realization
    print("Starting rainfall generation")
    storm_arrival_lambda=(1/9)*np.ones(num_patches)  # storm happens once per week
    storm_duration_gamma =24*np.ones(num_patches)  # storm lasts one hour
    cell_arrival_beta = 96*np.ones(num_patches)  # cell arrives every 15 minutes
    cell_duration_eta = 96*np.ones(num_patches)
    rainfall_intensity = 0*np.ones(num_patches)  # changed from original code

    if run_through:        
        # read in patch populations at county level and broken-up counties
        patch_pop = np.array(realworld_example["Patch Size"])
        avg_N = np.mean(patch_pop)

        # other disease parameters, given on day timescale
        multi_gamma = np.array([gamma]*num_patches)
        multi_delta = np.array([delta]*num_patches)
        multi_mu = np.array([mu]*num_patches)
        multi_c = np.array([c]*num_patches)
        multi_sigma = np.array([sigma]*num_patches)
        multi_omega = np.array([omega]*num_patches) 
        
        #Calculate scaled alpha array where larger populations get higher alpha values
        multi_alpha = alpha * (patch_pop / avg_N)
        
        # generate who-acquired-infection-from-whom matrix
        multi_beta = generating_beta_matrix(adjacent_matrix, beta, in_cluster_strength=0.5)
    
        # package disease params, run simulations
        disease_params = [multi_alpha,multi_beta,multi_mu,multi_c,multi_gamma,multi_delta,multi_omega,multi_sigma]
        abridged_disease_params = [multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_sigma, multi_omega]
        #sensitivities = detection_threshold*np.ones(num_patches)
        base_station_list = [0.0015]*50
        #base_station_list.extend([0.07]*25)
        sens_list = []

        for _ in range(0, orig_n_clusters):
            sens_list.extend(sample(base_station_list, k=min(len(base_station_list), num_patches)))
        if len(sens_list) < num_patches:
            sens_list.extend([0.0015] * (num_patches - len(sens_list)))
        sensitivities = np.array(sens_list[:num_patches])

        closure_file = os.path.join(save_folder_str, "closure_results")
        detection_days_file = os.path.join(save_folder_str, "first_detection_days.npy")
        detection_patches_file = os.path.join(save_folder_str, "first_detection_patches.npy")
        additional_alpha_file = os.path.join(save_folder_str, "additional_alpha_per_patch.npy")
        most_vax_cluster_file = os.path.join(save_folder_str, "most_vaccinated_cluster_per_closure.csv")

        # load pre-existing results, redefines some variables from above
        num_patches = len(clusters)
        closure_fracs = np.linspace(0.00, 0.99, 36)
        num_samples = 250
        #The sample number is how many times we simulate the disease spread for each scenario where we close some number of wastewater surveillance sites
        closure_scenario_case_counts = np.zeros(len(closure_fracs))
        all_detection_days = np.zeros((len(closure_fracs), num_samples))
        all_detection_patches = np.zeros((len(closure_fracs), num_samples))   
        
        # 3D array to store additional alpha for every [closure_frac_idx, sample_idx, patch_idx]
        all_additional_alphas = np.zeros((len(closure_fracs), num_samples, num_patches))
        most_vaccinated_clusters = []
        initial_operational_count = np.sum(realworld_operational_surveillance)
        
        for k,closure_frac in enumerate(closure_fracs):
            # how many sites are closed
            #operational_surveillance = random_site_closure(clusters, closure_frac)
            #extra_doses = np.sum(operational_surveillance)*site_to_dose_conversion
            print(f"\nNow working on fraction {100*closure_frac:.2f}")
            cumulative_case_counts = np.zeros(num_patches)

            #another modifcation from the original code to simply reduce test run times
            rainfall_matrix = rainfall_realization(storm_arrival_lambda, storm_duration_gamma, cell_arrival_beta, 
                                                   cell_duration_eta, rainfall_intensity, list(range(0, num_days)), 
                                                   np.eye(num_patches))
            
            # Track aggregate additional alpha allocated across all samples for this closure fraction
            closure_cluster_vax_sums = np.zeros(n_clusters)
            
            for i in range(0, num_samples):
                # choose initial disease location based on population
                initial_exponent = 1
                initial_prob = patch_pop**initial_exponent/(np.sum(patch_pop**initial_exponent))
                initial_exposure_patch = np.random.choice(num_patches, p=initial_prob)
                print(f"Sample number: {i}")
                
                #operational_surveillance = random_site_closure(clusters, closure_frac)
                operational_surveillance = sensitivity_site_closure(realworld_example, sensitivities, closure_frac)
                num_closed_sites = initial_operational_count - np.sum(operational_surveillance)
                extra_doses = num_closed_sites * site_to_dose_conversion
                print(f"number of closed sites:{num_closed_sites}")
                
                cumulative_case_counts_sample, clusters, det_day, det_patch, add_alpha = vaccination_strategy_better_wes_model(
                    tag, deepcopy(abridged_disease_params), patch_pop, extra_doses, rainfall_matrix, sensitivities=sensitivities, 
                    num_days=num_days, initial_cluster_allocation=1, operational_surveillance=operational_surveillance, detection_day_lag=detect_lag, 
                    chosen_patch=initial_exposure_patch, days_to_disperse_stockpile=stockpile_dispersal_days, multi_beta=multi_beta)
               
                all_detection_days[k, i] = det_day if det_day is not None else -1
                all_detection_patches[k, i] = det_patch if det_patch is not None else -1
                all_additional_alphas[k, i, :] = add_alpha
                
                for patch_idx in range(num_patches):
                    cluster_id = clusters[patch_idx]
                    closure_cluster_vax_sums[cluster_id] += add_alpha[patch_idx]
                
                cumulative_case_counts += cumulative_case_counts_sample
                
                if det_day is not None:
                    print(f" -> Disease detected on Day {det_day} in Patch {det_patch} (Cluster {clusters[det_patch]})")
                else:
                    print(" -> Disease went completely undetected.")

            cumulative_case_counts /= num_samples

            for j in range(0, num_patches):
                closure_scenario_case_counts[k] += cumulative_case_counts[j]

            top_cluster = np.argmax(closure_cluster_vax_sums)
            most_vaccinated_clusters.append({
                'closure_fraction': closure_frac,
                'most_vaccinated_cluster': top_cluster,
                'total_alpha_added_to_cluster': closure_cluster_vax_sums[top_cluster]
            })

        # save results to retrieve later
        np.save(closure_file, closure_scenario_case_counts)

        np.save(detection_days_file, all_detection_days)
        np.save(detection_patches_file, all_detection_patches)
        np.save(additional_alpha_file, all_additional_alphas)

        df_most_vax = pd.DataFrame(most_vaccinated_clusters)
        df_most_vax.to_csv(most_vax_cluster_file, index=False)

        min_cumlative = min(closure_scenario_case_counts)
        min_idx = np.argmin(closure_scenario_case_counts)
        min_frac = closure_fracs[min_idx]

        # administrative variables for graphing
        originating_cluster = clusters[initial_exposure_patch]
        labels = [f"{100*clos_frac:.2f}%" for clos_frac in closure_fracs]
        height = 1.25*max(closure_scenario_case_counts)
        bounding_factor = 1.23
        width=0.2

        colors = ['crimson' if i == min_idx else 'skyblue' for i in range(len(closure_fracs))]

        plt.bar(np.arange(len(closure_fracs)), closure_scenario_case_counts, width=width, color=colors)
        plt.xticks(np.arange(len(closure_fracs)), labels, rotation=290)
        plt.ylim([0, height])
        plt.xlabel("Site closure fraction")
        plt.ylabel("Cumulative case count")
        plt.title("Cumlative Case Count of real-life example test 3 beta 0.00001, RI=0")
        plt.figtext(0.30, 0.8, f"min frac:{min_frac:.4f}", 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.show()
