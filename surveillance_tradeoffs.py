from cluster_testing import *
from random import sample
from numpy.linalg import eig

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

# define rainfall patterns over course of simulation
# uses Poisson cluster model, with a matrix to distribute the effects of rainfall
# greek letters come from the paper I am using as a guide, may not be standard
# rates should be in days
def rainfall_realization(storm_arrival_lambda, storm_duration_gamma, cell_arrival_beta, cell_duration_eta, rainfall_intensity,
                          t, geo_connectivity):
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

    # need to make more efficient if whole method is slow
    for day_index, day in enumerate(t):
        for raincell in cell_list:
            if raincell.start_time <= day <= raincell.end_time:
                rainfall[raincell.storm_origin, day_index] += raincell.intensity
    
    return np.matmul(geo_connectivity, rainfall)  # take geographic structure into account

# E, I, N are patches x timesteps
def extensive_wes_data(E, I, N, t, gamma, R_0, V_p, rainfall, kappa=None, rho_max=1):
    if kappa is None:
        kappa = -gamma*np.log(1-0.5)

    # when do we count the disease as appearing in a patch?
    infection_thresholds = I>1
    disease_arrival_days = np.array([t[index] for index in np.apply_along_axis(find_true,0,infection_thresholds)])  # disease has arrived first day there is 1 whole infection
    
    shedding_numerator = ((1-np.exp(-R_0*(max(t-disease_arrival_days,0))))*(1-gamma/(gamma+kappa))*E+I)*rho_max*V_p
    total_water_denominator = N*V_p+rainfall
    shedding_fractions = shedding_numerator/total_water_denominator
    
    return np.array(shedding_fractions)

def sensitivity_site_closure(clusters, sensitivities, closure_perc):
    # extract total patch numbers and cluster information
    num_patches = len(clusters)
    n_clusters = len(set(clusters))

    total_sensitivities = np.zeros(n_clusters)
    # default to all stations being open
    operational_surveillance = [True]*num_patches
    cluster_prob_dicts = [{} for _ in range(0, n_clusters)]
    
    for i in range(0,num_patches):
        total_sensitivities[clusters[i]] += sensitivities[i]
    
    # add normalized probability for each cluster
    for i in range(0, num_patches):
        cluster_prob_dicts[clusters[i]][i] = sensitivities[i]/total_sensitivities[clusters[i]]
    
    for j in range(0,n_clusters):
        # how many surveillance stations to close
        num_stations_to_close = round(closure_perc*len(cluster_prob_dicts[j]))
        stations_closed = np.random.choice(list(cluster_prob_dicts[j].keys()), size=num_stations_to_close,
                                            replace=False, p=list(cluster_prob_dicts[j].values()))
        # mark each selected station as closed
        for station in stations_closed:
            operational_surveillance[station] = False
    
    return operational_surveillance

def vaccination_strategy_extra_doses(tag, abridged_disease_params, patch_populations, extra_doses_per_day, unif_low=0.6, unif_high=1, sensitivities=0.03, 
                         num_days=1500, scale_factor=8.72*np.log(10),
                         chosen_patch=0, initial_cluster_allocation=1, operational_surveillance=None, detection_day_lag=0,
                         days_to_disperse_stockpile=0):
    
    # find files to load data
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")

    # load pre-existing results
    clusters = np.load(clusters_file)
    n_clusters = len(set(clusters))  # how many non-duplicated cluster labels are present
    num_patches = len(clusters)
    multi_beta = np.load(beta_file)

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

    # combine preloaded beta with other disease parameters
    multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta = abridged_disease_params
    disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta]
    
    # prepapre the initial state for the simulation
    pop_state = np.zeros(5*num_patches)
    pop_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
    pop_state[num_patches:2*num_patches] = (multi_alpha/(multi_mu+multi_alpha))*patch_populations
    frac_exposed = 0.01*pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[2*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state
    track_cumulative_totals = np.zeros(num_patches)
    
    # track which clusters have currently registered a detection event
    detections_in_cluster = [False]*n_clusters
    sia_intervention_allocated = False
    stockpile_days = None  # need to do this to avoid non-assignment area later down

    for day in range(0, num_days):
        # simulate one day of disease evolution
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1,:]
        susceptible = pop_state[:num_patches]  # may use to allocate vaccine budget, redundant for now
        exposures = pop_state[2*num_patches:3*num_patches]
        infected = pop_state[3*num_patches:4*num_patches]

        # we get one sia per simulation, check if it has been used
        if not sia_intervention_allocated:
            # generate wes data and check for detections
            shedding = exposures + infected
            wes_data = [shedding[i]*scale_factor*rng.uniform(low=unif_low, high=unif_high) for i in range(0, num_patches)]
            # detections modified to ensure site has been selected to continue functioning
            potential_detections = [(wes >= patch_populations[j]*sensitivities[j]*scale_factor) and operational_surveillance[j] for j,wes in enumerate(wes_data)]

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
                    else:
                        additional_alpha = other_clusters_budget/non_origin_total_pop
                        multi_alpha[j] += additional_alpha
            
            # do this at the end so that of this if statement so that a lag of zero actually means zero
            countdown_to_vaccination -= 1
        
        # we get to use our built-up stockpile of vaccines over a very short number of days, unless we have disabled this mode
        # by setting the number of days to disperse our stockpile to zero
        if (stockpile_days is not None) and stockpile_days > 0:
            stockpile_days -= 1
            for j in range(0, num_patches):
                if clusters[j] == ground_zero_cluster:
                    susceptible[j] -= min(stockpiled_per_patch_per_day, susceptible[j])
    
    track_cumulative_totals = pop_state[4*num_patches:]
    return track_cumulative_totals, clusters

# maybe make rho_max 4.36*np.log(10)
def vaccination_strategy_better_wes_model(tag, abridged_disease_params, patch_populations, extra_doses_per_day, rainfall_matrix,
                                           sensitivities=0.03, num_days=1500, rho_max=1, V_p=1,
                                           chosen_patch=0, initial_cluster_allocation=1, operational_surveillance=None, detection_day_lag=0,
                                           days_to_disperse_stockpile=0):
    
    # find files to load data
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")

    # load pre-existing results
    clusters = np.load(clusters_file)
    n_clusters = len(set(clusters))  # how many non-duplicated cluster labels are present
    num_patches = len(clusters)
    multi_beta = np.load(beta_file)

    # compute spectral radius
    eigens, _ = eig(multi_beta)
    spec_beta = max(abs(eigens))

    # important to ensure this is a numpy array
    patch_populations = np.array(patch_populations)
    kappa = -gamma*np.log(1-0.5)

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

    # combine preloaded beta with other disease parameters
    multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta = abridged_disease_params
    disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta]

    # calculate basic reproduction number
    R_0 = np.average(patch_populations)*spec_beta*(multi_c[0]*(multi_mu[0]+multi_delta[0])+multi_gamma[0])/((multi_mu[0]+multi_delta[0])*(multi_mu[0]+multi_gamma[0]))
    #print(f"Basic reproduction number: {R_0:.3f}")
    
    # prepapre the initial state for the simulation
    pop_state = np.zeros(5*num_patches)
    pop_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
    pop_state[num_patches:2*num_patches] = (multi_alpha/(multi_mu+multi_alpha))*patch_populations
    frac_exposed = 0.01*pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[2*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state
    track_cumulative_totals = np.zeros(num_patches)
    
    # track which clusters have currently registered a detection event
    detections_in_cluster = [False]*n_clusters
    sia_intervention_allocated = False
    stockpile_days = None  # need to do this to avoid non-assignment area later down
    evident_infections_date = np.array([np.inf]*num_patches)

    for day in range(0, num_days):
        # simulate one day of disease evolution
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1,:]
        susceptible = pop_state[:num_patches]  # may use to allocate vaccine budget, redundant for now
        exposures = pop_state[2*num_patches:3*num_patches]
        infected = pop_state[3*num_patches:4*num_patches]
        
        for i,date in enumerate(evident_infections_date):
            if np.isfinite(date) and infected[i]>=1:
                evident_infections_date[i] = day

        # we get one sia per simulation, check if it has been used
        if not sia_intervention_allocated:
            # generate wes data and check for detections
            wes_numerator = ((1-np.exp(-R_0*(np.maximum(day-evident_infections_date,0))))*(1-gamma/(gamma+kappa))*exposures+infected)*rho_max*V_p
            wes_data = wes_numerator/(patch_populations*V_p+rainfall_matrix[:,day])
            # detections modified to ensure site has been selected to continue functioning
            potential_detections = [(wes >= sensitivities[j]*rho_max) and operational_surveillance[j] for j,wes in enumerate(wes_data)]

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
                    else:
                        additional_alpha = other_clusters_budget/non_origin_total_pop
                        multi_alpha[j] += additional_alpha
            
            # do this at the end so that of this if statement so that a lag of zero actually means zero
            countdown_to_vaccination -= 1
        
        # we get to use our built-up stockpile of vaccines over a very short number of days, unless we have disabled this mode
        # by setting the number of days to disperse our stockpile to zero
        if (stockpile_days is not None) and stockpile_days > 0:
            stockpile_days -= 1
            for j in range(0, num_patches):
                if clusters[j] == ground_zero_cluster:
                    susceptible[j] -= min(stockpiled_per_patch_per_day, susceptible[j])
    
    track_cumulative_totals = pop_state[4*num_patches:]
    return track_cumulative_totals, clusters

if __name__ == "__main__":
    tag = "debugging_improved_wes_model"
    run_through = True
    cluster_first = True

    # patch connection strengths
    in_patch = 10**(-2)
    out_patch = in_patch*10**(-1)

    # cluster information
    cluster_sizes = [50]*5  # 250 catchment sites in 5 subgroups
    num_patches = np.sum(cluster_sizes)
    orig_n_clusters = len(cluster_sizes)
    n_clusters = 5*orig_n_clusters  # try extra clusters

    # other disease parameters, given on day timescale
    gamma = np.log(1/0.9)/7
    delta = np.log(1/0.9)/10
    mu = 0.01
    c = 0.2
    alpha = 0.03
    beta = 0.3/1000

    # for monitoring
    num_days = 600
    detection_threshold = 0.07
    site_to_dose_conversion = 70.29
    detect_lag=4
    initial_exposure_patch = 0
    stockpile_dispersal_days = 0

    # calculate a rainfall realization
    print("Starting rainfall generation")
    storm_arrival_lambda=1/3*np.ones(num_patches)  # storm happens once per week
    storm_duration_gamma =24*np.ones(num_patches)  # storm lasts one hour
    cell_arrival_beta = 96*np.ones(num_patches)  # cell arrives every 15 minutes
    cell_duration_eta = 96*np.ones(num_patches)
    rainfall_intensity = 1000*np.array([rng.uniform() for _ in range(0,num_patches)])  # just for testing purposes

    one_patch_number = basic_reproduction_number(alpha, beta, mu, c, gamma, delta, N=1000)
    print(f"R_0 in isolated patch: {one_patch_number}")

    if run_through:
        # other disease parameters, given on day timescale
        multi_gamma = np.array([gamma]*num_patches)
        multi_delta = np.array([delta]*num_patches)
        multi_mu = np.array([mu]*num_patches)
        multi_c = np.array([c]*num_patches)
        multi_alpha = np.array([alpha]*num_patches)
        
        # construct a transmission matrix
        multi_beta = transmission_matrix(beta, cluster_sizes=cluster_sizes, in_cluster_strength=in_patch, out_cluster_strength=out_patch,
                                connect_to_frac=0.5)
        
        # package disease params, run simulations
        disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta]
        abridged_disease_params = [multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta]
        #sensitivities = detection_threshold*np.ones(num_patches)
        base_station_list = [0.0015]*50
        #base_station_list.extend([0.07]*25)
        sens_list = []
        for _ in range(0, orig_n_clusters):
            sens_list.extend(sample(base_station_list, k=len(base_station_list)))
        sensitivities = np.array(sens_list)
        
        #patch_pop = 1000*np.ones(num_patches)
        big_pops = [5000,5000,5000,5000,5000]
        big_pops.extend([550]*45)
        patch_pop = np.array(big_pops*5)
        if cluster_first:
            clustering_scenario(tag, disease_params, patch_pop, cluster_sizes, unif_low=0.6, unif_high=1,
                                    sensitivities=sensitivities, num_days=num_days, detection_day_lag=detect_lag)

        # generate plots from clustering
        cwd = os.getcwd()
        save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
        clusters_file = os.path.join(save_folder, "clusters.npy")
        beta_file = os.path.join(save_folder, "beta.npy")
        detect_file = os.path.join(save_folder, "detection_times.npy")

        # create a directory to save different closure results
        closure_file = os.path.join(save_folder, "closure_results")

        # load pre-existing results, redefines some variables from above
        clusters = np.load(clusters_file)
        all_detection_times = np.load(detect_file)
        n_clusters = len(set(clusters))  # how many non-duplicated cluster labels are present
        num_patches = len(clusters)
        multi_beta = np.load(beta_file)

        #cluster_accuracy(clusters, [50,50,50,50,50])
        #clustering_charts(all_detection_times, cluster_sizes, clusters)

        closure_fracs = np.linspace(0, 0.99, 36)
        num_samples = 250
        closure_scenario_case_counts = np.zeros(len(closure_fracs))
        
        for k,closure_frac in enumerate(closure_fracs):
            # how many sites are closed
            #operational_surveillance = random_site_closure(clusters, closure_frac)
            #extra_doses = np.sum(operational_surveillance)*site_to_dose_conversion
            print(f"\nNow working on fraction {100*closure_frac:.2f}")
            cumulative_case_counts = np.zeros(num_patches)

            for i in range(0, num_samples):
                print(f"Sample number: {i}")
                if i%5 == 0:
                    rainfall_matrix = rainfall_realization(storm_arrival_lambda, storm_duration_gamma, cell_arrival_beta, cell_duration_eta, rainfall_intensity, list(range(0,num_days)), np.eye(num_patches))
                    print("New rainfall generated")
                #operational_surveillance = random_site_closure(clusters, closure_frac)
                operational_surveillance = sensitivity_site_closure(clusters, sensitivities, closure_frac)
                extra_doses = (num_patches-np.sum(operational_surveillance))*site_to_dose_conversion
                cumulative_case_counts_sample, clusters = vaccination_strategy_better_wes_model(tag, deepcopy(abridged_disease_params), patch_pop, 
                                                                                           extra_doses, rainfall_matrix, sensitivities=sensitivities, num_days=num_days,
                                                                                           initial_cluster_allocation=1, operational_surveillance=operational_surveillance,
                                                                                           detection_day_lag=detect_lag, chosen_patch=initial_exposure_patch,
                                                                                           days_to_disperse_stockpile=stockpile_dispersal_days)
                cumulative_case_counts += cumulative_case_counts_sample

            cumulative_case_counts /= num_samples
            # for j in range(0, num_patches):
            #     if clusters[j] == clusters[initial_exposure_patch]:
            #         closure_scenario_case_counts[k] += cumulative_case_counts[j]

            for j in range(0, num_patches):
                closure_scenario_case_counts[k] += cumulative_case_counts[j]
            
        # save results to retrieve later
        np.save(closure_file, closure_scenario_case_counts)

        # administrative variables for graphing
        originating_cluster = clusters[initial_exposure_patch]
        labels = [f"{100*clos_frac:.2f}%" for clos_frac in closure_fracs]
        height = 1.25*max(closure_scenario_case_counts)
        bounding_factor = 1.23
        width=0.2

        plt.bar(np.arange(len(closure_fracs)), closure_scenario_case_counts, width=width)
        plt.xticks(np.arange(len(closure_fracs)), labels, rotation=290)
        plt.ylim([0, height])
        plt.xlabel("Site closure fraction")
        plt.ylabel("Cumulative case count")
        plt.show()
