from cluster_testing_Timas_scipy_copy import *
from random import sample
from numpy.linalg import eig
import random
from pathlib import Path
import numpy as np
import pandas as pd
import math
import time

random.seed(42) #not needed but added to compare with other code templates
 
rng = np.random.default_rng()

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

print("[SETUP] Starting file loading and preprocessing...")
setup_start = time.perf_counter()

file_name = "kentucky-counties-by-population-(2026).xlsx"
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

adjacent_matrix = pd.read_csv(
    "/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/adjacent weighted matrix.csv")

adjacent_matrix = adjacent_matrix.set_index("Unnamed: 0")

print(f"[SETUP] Task completed in {time.perf_counter() - setup_start:.4f} seconds.")

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

    main_diag = np.diag(multi_beta)

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
    save_folder = script_dir / "test 2" / tag

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
    
    #print(f"Basic reproduction number: {R_0:.3f}")
    
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
    
    avg_ode_time = np.mean(ode_daily_times)
    median_ode_time = np.median(ode_daily_times)
    mode_ode_time = stats.mode(ode_daily_times, keepdims=True).mode[0]
    std_ode_time = np.std(ode_daily_times)
    max_ode_time = np.max(ode_daily_times)
    min_ode_time = np.min(ode_daily_times)

    print("\n    " + "="*55)
    print("    [ODE SOLVER RUNTIME SUMMARY STATISTICS]")
    print(f"      Average (Mean):     {avg_ode_time:.6f} s")
    print(f"      Median:             {median_ode_time:.6f} s")
    print(f"      Mode:               {mode_ode_time:.6f} s")
    print(f"      Standard Deviation: {std_ode_time:.6f} s")
    print(f"      Maximum:            {max_ode_time:.6f} s")
    print(f"      Minimum:            {min_ode_time:.6f} s")
    print("    " + "="*55 + "\n")
    
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

    script_dir = Path(__file__).resolve().parent
    save_folder = script_dir / "test 2" / tag
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
    beta = 0.15/10000 #how likely is an infected person to spread the disease to someone else
    omega = 0.0001
    sigma=0.0001

    # for monitoring
    num_days = 600
    detection_threshold = 0.07
    site_to_dose_conversion = 70.29
    detect_lag=4
    initial_exposure_patch = rng.choice(range(0, len(realworld_example)-1))
    stockpile_dispersal_days = 0

    # calculate a rainfall realization
    print("Starting rainfall generation")
    storm_arrival_lambda=(1/9)*np.ones(num_patches)  # storm happens once per week
    storm_duration_gamma =24*np.ones(num_patches)  # storm lasts one hour
    cell_arrival_beta = 96*np.ones(num_patches)  # cell arrives every 15 minutes
    cell_duration_eta = 96*np.ones(num_patches)
    rainfall_intensity = 52500.52*np.ones(num_patches)  # changed from original code

    if run_through:        
        
        patch_pop = np.array(realworld_example["Patch Size"])
        
        avg_N = np.mean(patch_pop)
        # other disease parameters, given on day timescale
        multi_gamma = np.array([gamma]*num_patches)
        multi_delta = np.array([delta]*num_patches)
        multi_mu = np.array([mu]*num_patches)
        multi_c = np.array([c]*num_patches)
        
        #Calculate scaled alpha array where larger populations get higher alpha values
        multi_alpha = alpha * (patch_pop / avg_N)
        
        multi_sigma = np.array([sigma]*num_patches)
        multi_omega = np.array([omega]*num_patches) 
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
        
        # generate plots from clustering
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
                print(f"Sample number: {i}")
                
                #operational_surveillance = random_site_closure(clusters, closure_frac)
                operational_surveillance = sensitivity_site_closure(realworld_example, sensitivities, closure_frac)

                extra_doses = (num_patches-np.sum(operational_surveillance))*site_to_dose_conversion
               
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
        

        np.save(additional_alpha_file, all_additional_alphas)

        df_most_vax = pd.DataFrame(most_vaccinated_clusters)
        df_most_vax.to_csv(most_vax_cluster_file, index=False)

        min_cumlative=min(closure_scenario_case_counts)
        min_frac= closure_fracs[np.where(closure_scenario_case_counts==min_cumlative)]

        # administrative variables for graphing
        originating_cluster = clusters[initial_exposure_patch]
        labels = [f"{100*clos_frac:.2f}%" for clos_frac in closure_fracs]
        height = 1.25*max(closure_scenario_case_counts)
        bounding_factor = 1.23
        width=0.2

        colors = ['crimson' if i == int(np.where(closure_scenario_case_counts==min_cumlative)[0]) else 'skyblue' for i in range(len(closure_fracs))]

        plt.bar(np.arange(len(closure_fracs)), closure_scenario_case_counts, width=width, color=colors)
        plt.xticks(np.arange(len(closure_fracs)), labels, rotation=290)
        plt.ylim([0, height])
        plt.xlabel("Site closure fraction")
        plt.ylabel("Cumulative case count")
        plt.title("Cumalyive Case Count of real-life example")
        plt.figtext(0.30, 0.8, f"min frac:{min_frac[0]}", 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.show()
#%% Generating a time check plot for rainfall realization 

import time

#making a test parameter list using patch numbers
patch_list = list(range(num_patches+1)) 
patch_list= patch_list[1:]

execution_times = []

for patch in patch_list:
    geo_conn_patch = np.eye(patch)

#splicing the list of definition input to make sure its the same size of the patch size at each test run
    lambda_patch = storm_arrival_lambda[:patch]
    gamma_patch = storm_duration_gamma[:patch]
    beta_patch = cell_arrival_beta[:patch]
    eta_patch = cell_duration_eta[:patch]
    intensity_patch = rainfall_intensity[:patch]

    start_time = time.perf_counter() 
    
    rainfall_realization(
        lambda_patch, 
        gamma_patch, 
        beta_patch, 
        eta_patch, 
        intensity_patch, 
        list(range(0, num_days)), 
        geo_conn_patch
    )
    
    end_time = time.perf_counter()
    execution_times.append(end_time - start_time)
    
plt.figure(figsize=(8, 5))
plt.plot(patch_list, execution_times, marker='o', linestyle='-', color='b', label='Execution Time')

plt.title("Time Complexity: Rain Realization vs Patch Number after mod1")
plt.xlabel("Patch Number")
plt.ylabel("Execution Time (Seconds)")
plt.grid(True)
plt.legend()
plt.show()
#%%
O_0_1_day=np.load("/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/sigma=0.00001, omega=0.1 data/debugging_improved_wes_model/first_detection_days.npy")
O_0_5_day=np.load("/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/sigma=0.00001, omega=0.5 data/debugging_improved_wes_model/first_detection_days.npy")

O_0_1_patch=np.load("/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/sigma=0.00001, omega=0.1 data/debugging_improved_wes_model/first_detection_patches.npy")
O_0_5_patch=np.load("/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/sigma=0.00001, omega=0.5 data/debugging_improved_wes_model/first_detection_patches.npy")

def hist_day_and_patch(day,patch,intensity):
    
    closure_fracs = np.linspace(0.60, 0.85, 36)

    rows, cols = day.shape
    
    combined_matrix = np.empty((rows, 2 * cols), dtype=day.dtype)
    combined_matrix[:, 0::2] = day
    combined_matrix[:, 1::2] = patch
    
    col_names = []
    for i in range(1, num_samples + 1):
        col_names.append(f"RI {intensity} day_sample {i}")
        col_names.append(f"RI {intensity} patch_sample {i}")

    row_names = np.array([f"fraction_{frac:.4f}" for frac in closure_fracs])
    row_names = np.insert(row_names,0,"closure fraction")
    
    combined_matrix= np.vstack((col_names, combined_matrix))
    combined_matrix=np.column_stack((row_names,combined_matrix))
    
    np.savetxt(f"RI {intensity} zoom in days and patches.csv", combined_matrix, delimiter=",", fmt="%s")
    
    target_frac = 0.72857
    
    frac_idx = np.where(np.isclose(closure_fracs, target_frac))[0][0]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharey=True)

    axes[0, 0].hist(day[0], bins=25, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlim(-1,300)
    axes[0, 0].set_title(f"sigma 0.00005 and omega {intensity} first infection day frac 0.0")
    axes[0, 0].set_xlabel('day')
    
    axes[0, 1].hist(patch[0], bins=25, color='salmon', edgecolor='black')
    axes[0, 1].set_xlim(-1,300)
    axes[0, 1].set_title(f"sigma 0.00005 and omega {intensity} first infection patch frac 0.0")
    axes[0, 1].set_xlabel('patch')
    
    axes[1, 0].hist(day[35], bins=25, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlim(-1,300)
    axes[1, 0].set_title(f"sigma 0.00005 and omega {intensity} first infection day frac 0.99")
    axes[1, 0].set_xlabel('day')
    
    axes[1, 1].hist(patch[35], bins=25, color='salmon', edgecolor='black')
    axes[1, 1].set_xlim(-1,300)
    axes[1, 1].set_title(f"sigma 0.00005 and omega {intensity} first infection patch frac 0.99")
    axes[1, 1].set_xlabel('patch')

    axes[2, 0].hist(day[frac_idx], bins=25, color='skyblue', edgecolor='black')
    axes[2, 0].set_xlim(-1,300)
    axes[2, 0].set_title(f"sigma 0.00005 and omega {intensity} first infection day frac min ({target_frac:.4f})")
    axes[2, 0].set_xlabel('day')
    
    axes[2, 1].hist(patch[frac_idx], bins=25, color='salmon', edgecolor='black')
    axes[2, 1].set_xlim(-1,300)
    axes[2, 1].set_title(f"sigma 0.00005 and omega {intensity} first infection patch frac min ({target_frac:.4f})")
    axes[2, 1].set_xlabel('patch')

    plt.tight_layout()
    plt.show()
    
hist_day_and_patch(O_0_5_day, O_0_5_patch, 0.5)   

#%% Creating a table of most common first day detection days

RI_500=pd.read_csv("/Users/timaa/Desktop/OHT 2026/surveillance trade offs python/RI 500 days and patches.csv", index_col=0)

def combine_find_max_pair(df, intensity):
    combined_df = pd.DataFrame(index=df.index)
    
    for i in range(0, len(df.columns), 2):
        col1 = df.columns[i]
        col2 = df.columns[i + 1]
    
        new_col_name = col1.strip()
    
        combined_df[new_col_name] = (
            "(" + df[col1].astype(str) + ", " + df[col2].astype(str) + ")"
        ) 
        
    column_names = [f"Sample{i}" for i in range(1, 250 + 1)]
    
    combined_df.columns = column_names
    
    combined_df.to_csv(f"combined RI {intensity}.csv")
    
    most_common_pairs = []
    highest_frequencies = []

    for index, row in combined_df.iterrows():

        row_counts = row.value_counts()

        most_common_pairs.append(row_counts.index[0])

        highest_frequencies.append(row_counts.iloc[0])

    return np.array(most_common_pairs), np.array(highest_frequencies)  

RI_500_most_common_pair, RI_500_highest_frequency= combine_find_max_pair(RI_500, 500)

day_and_patch_most_common=[[RI_50_most_common_pair, RI_100_most_common_pair, RI_125_most_common_pair, RI_150_most_common_pair, RI_175_most_common_pair, RI_200_most_common_pair],
                                   [RI_50_highest_frequency, RI_100_highest_frequency, RI_125_highest_frequency, RI_150_highest_frequency, RI_175_highest_frequency, RI_200_highest_frequency]]

labels = RI_500_most_common_pair
counts = RI_500_highest_frequency

df = pd.DataFrame({
    'Day and Patch (Most Common)': labels,
    'Frequency / Count': counts,
})

row_names=closure_fracs
row_names = np.round(row_names, 4)

df.index=row_names

df.to_csv("most common day and patch first infection RI 500.csv")
#%%

max_vac_clusters = np.load("most_vaccinated_cluster_per_sample.npy")  # Replace with your actual path


def max_vax_cluster_hist(clusters, n_clusters):
    closure_fracs = np.linspace(0.0, 0.99, 36)

    target_frac = 0.72857
    frac_idx = np.where(np.isclose(closure_fracs, target_frac))[0][0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

    bins = np.arange(-0.5, n_clusters + 0.5, 1)

    axes[0].hist(clusters[0], bins=bins, color="skyblue", edgecolor="black")
    axes[0].set_title(f"Kentucky Clusters with Max Vaccination (Closure Frac {closure_fracs[0]:.2f})")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(range(n_clusters))

    axes[1].hist(clusters[35], bins=bins, color="skyblue", edgecolor="black")
    axes[1].set_title(f"Kentucky Clusters with Max Vaccination (Closure Frac {closure_fracs[35]:.2f})")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xticks(range(n_clusters))

    axes[2].hist(
        clusters[frac_idx], bins=bins, color="skyblue", edgecolor="black"
    )
    axes[2].set_title(f"Kentucky Clusters with Max Vaccination (Closure Frac {closure_fracs[frac_idx]:.4f})")
    axes[2].set_xlabel("Cluster ID")
    axes[2].set_ylabel("Frequency")
    axes[2].set_xticks(range(n_clusters))

    plt.tight_layout()
    plt.show()

max_vax_cluster_hist(max_vac_clusters, n_clusters=orig_n_clusters)
#%%

all_additional_alphas = np.load(
    "test/debugging_improved_wes_model/additional_alpha_per_patch.npy"
)

closure_fracs = np.linspace(0.00, 0.99, 36)
target_frac = 0.72857
frac_idx = np.where(np.isclose(closure_fracs, target_frac))[0][0]

indices_to_compare = [0, frac_idx, 35]  
labels = [
    f"Frac {closure_fracs[0]:.2f}",
    f"Frac {closure_fracs[frac_idx]:.4f}",
    f"Frac {closure_fracs[35]:.2f}",
]
colors = ["skyblue", "salmon", "mediumpurple"]

plt.figure(figsize=(16, 6))
width = 0.25

for i, idx in enumerate(indices_to_compare):
    mean_alpha = np.mean(all_additional_alphas[idx, :, :], axis=0)
    plt.bar(
        patch_indices + (i * width),
        mean_alpha,
        width=width,
        label=labels[i],
        color=colors[i],
        edgecolor="black",
        alpha=0.85,
    )

plt.xlabel("Patch ID", fontsize=12)
plt.ylabel(r"Average Additional $\alpha$", fontsize=12)
plt.title(
    r"Average Change in $\alpha$ per Patch at Selected Closure Fractions",
    fontsize=14,
)
plt.xticks(patch_indices + width, patch_indices, rotation=90, fontsize=8)
plt.legend(title="Closure Fraction")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()