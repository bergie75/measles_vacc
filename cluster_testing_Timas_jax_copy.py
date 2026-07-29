import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as colormap
from dtw import dtw
import os
import time
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as jnp

rng = np.random.default_rng()

def find_detection_times(wes_data, timepoints, sensitivities, scale_factor=8.72*np.log(10), N=None, detection_day_lag=0):
    if N is None:
        num_patches = len(sensitivities)
        N=np.ones(num_patches)
    
    # perform stacking for compatibility with previous code, multiply population into sensitivities 
    num_timepoints = len(timepoints)
    scaled_sensitivities = np.vstack([sensitivities*N]*num_timepoints)
    
    # a vector to hold the detection times
    detection_times = []

    # precursor to detection times
    detections = (wes_data >= (scaled_sensitivities*scale_factor))

    for j in range(0, wes_data.shape[1]):
        for i, time in enumerate(timepoints):
            if detections[i,j] or i==len(timepoints)-1:
                detection_times.append(time+detection_day_lag)
                break
    
    return np.array(detection_times)

def basic_reproduction_number(alpha, beta, mu, c, gamma, delta, N):
    return N*beta*mu/(mu+alpha)*(c*(mu+delta)+gamma)/((mu+gamma)*(mu+delta))

def compute_total_daily_doses(num_wes_closed, base_daily_doses, cost_per_dose=95.201, daily_wes_cost=6691.3):
    doses_per_closed_wes = daily_wes_cost/cost_per_dose
    return base_daily_doses + num_wes_closed*doses_per_closed_wes

def create_wes_data(exposure_timeseries, scale_factor=8.72*np.log(10), unif_low=1, unif_high=1, minimum=0):
    perfect_data = exposure_timeseries*scale_factor

    measurement_outputs = []
    for result in perfect_data:
        messy_result = result*rng.uniform(low=unif_low, high=unif_high)
        if messy_result > minimum:
            measurement_outputs.append(messy_result)
        else:
            measurement_outputs.append(0)
    
    return np.array(measurement_outputs)

def random_site_closure(clusters, closure_perc):
    # extract total patch numbers and cluster information
    num_patches = len(clusters)

    # vector which determines whether site results should be considered
    # on average, this closes closure_perc of the sites in each cluster
    operational_surveillance = [rng.uniform() >= closure_perc for _ in range(0, num_patches)]
    
    return operational_surveillance

def transmission_matrix(raw_beta, cluster_sizes, in_cluster_strength=0.5, out_cluster_strength=0.01, connect_to_frac=1):
    # recreate number of patches
    num_patches = int(np.sum(cluster_sizes))
    
    # we will recursively grow this array
    upper_triangle = np.zeros((num_patches, num_patches))
    starting_index = 0

    # create rows one by one, transpose for symmetry
    for cluster_size in cluster_sizes:
        for i in range(starting_index, starting_index+cluster_size):
            # in cluster entries, on the block diagonal
            for j in range(i+1, starting_index+cluster_size):
                u = rng.uniform()
                if u >= 1-connect_to_frac:
                    upper_triangle[i,j] = in_cluster_strength*rng.uniform()
            # out of cluster entries that come after the block diagonal
            for j in range(starting_index+cluster_size, num_patches):
                u = rng.uniform()
                if u >= 1-connect_to_frac:
                    upper_triangle[i,j] = out_cluster_strength*rng.uniform()
        
        starting_index += cluster_size
    
    waifw_matrix = raw_beta*(np.eye(num_patches) + upper_triangle + np.transpose(upper_triangle))
    return waifw_matrix

def transmission_matrix_realworld_inter(raw_beta, adjacent_matrix, in_cluster_strength=0.5, out_cluster_strength=0.01, connect_to_frac=1):
    # recreate number of patches
    num_patches = len(adjacent_matrix)
    cluster_sizes= adjacent_matrix['Cluster'].value_counts() 
    # we will recursively grow this array
    upper_triangle = np.zeros((num_patches, num_patches))
    starting_index = 0

    # create rows one by one, transpose for symmetry
    for cluster_size in cluster_sizes:
        for i in range(starting_index, starting_index+cluster_size):
            # in cluster entries, on the block diagonal
            for j in range(i+1, starting_index+cluster_size):
                u = rng.uniform()
                if u >= 1-connect_to_frac:
                    upper_triangle[i,j] = in_cluster_strength*rng.uniform()
            # out of cluster entries that come after the block diagonal
            for j in range(starting_index+cluster_size, num_patches):
                u = rng.uniform()
                if u >= 1-connect_to_frac:
                    upper_triangle[i,j] = out_cluster_strength*rng.uniform()
        
        starting_index += cluster_size
    
    waifw_matrix = raw_beta*(np.eye(num_patches) + upper_triangle + np.transpose(upper_triangle))
    return waifw_matrix

def compartment_rhs_multi_patch_jax(x, t, disease_params):
    
    num_patches = x.shape[0] // 6
    
    # Reshape 1D tracer 'x' back to 2D matrix (6, num_patches)
    x = x.reshape((6, num_patches))
    
    # break state up into epidemiologically relevant categories
    #x is all subpopulations
    S1= x[0, :]
    S2= x[1, :]
    V= x[2, :]
    E= x[3, :]
    I= x[4, :]
    cum_cases = x[5, :]

    # omega: rate of losing vaccine immunity (V -> S2)
    # sigma: relative susceptibility/breakthrough factor for vaccinated individuals
    alpha, beta, mu, c, gamma, delta, omega, sigma, N = disease_params

    N = jnp.ones(num_patches)
    
    force_of_infection = jnp.matmul(beta,(c*E+I))

    dS1_dt = mu*(N-S1)-alpha*S1-force_of_infection*S1 #mu*(N-S) represents birth minus death rate
    dS2_dt = omega*V - force_of_infection*S2 - mu*S2    #beta is a big matrix that specifies how patches are interacting
    dV_dt = alpha * S1 - omega * V - sigma * force_of_infection * V - mu * V                    # with each other, mu*V are people dying while vaccinated            
    dE_dt = force_of_infection*(S1+S2+sigma*V)    
    dI_dt = gamma*E-(mu+delta)*I
    dcum_cases_dt = I #keep tracking of the cumulative case count
    
    return jnp.stack([dS1_dt, dS2_dt, dV_dt, dE_dt, dI_dt, dcum_cases_dt], axis=0).ravel()


def clustering_scenario(tag, disease_params, patch_populations, cluster_sizes, unif_low=0.6, unif_high=1, sensitivities=0.03, num_days=1500,
                        minimum=0, scale_factor=8.72*np.log(10), detection_day_lag=0):
    # unpack useful variables
    multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma= disease_params
    num_patches = len(patch_populations)
    num_timepoints=num_days+1
    timepoints = np.linspace(0,num_days,num_timepoints)
    all_detection_times = np.zeros((num_patches, num_patches))
    
    # expand test sensitivities. If scalar given, all sensitivities are the same. If vector, then sensitivity varies by catchment
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = sensitivities*np.ones(num_patches)
    
    # start all catchment areas at disease free equilibrium vaccination level
    for chosen_patch in range(0, num_patches):
        print(f"Computing patch {chosen_patch}")
        initial_state = np.zeros(6*num_patches)
        initial_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
        frac_exposed = 0.01*initial_state[chosen_patch]
        initial_state[chosen_patch] -= frac_exposed
        initial_state[3*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state

        disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches, patch_populations))
        exposures = disease_sim[:,3*num_patches:4*num_patches]
        infected = disease_sim[:,4*num_patches:5*num_patches]

        # save results of wastewater
        wes_data = np.zeros(exposures.shape)
        for i in range(0, num_patches):
            wes_data[:,i] = create_wes_data(exposures[:,i]+infected[:,i], unif_low=unif_low, unif_high=unif_high, minimum=minimum,
                                             scale_factor=scale_factor)
        
        # compute when disease is actually found in wastewater
        all_detection_times[chosen_patch,:] = find_detection_times(wes_data, timepoints, sensitivities, 
                                                                   scale_factor=scale_factor, N=patch_populations, detection_day_lag=detection_day_lag)

    # create save directory
    script_dir = Path(__file__).resolve().parent
    save_folder = script_dir / "test" / tag
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # save found clusters, detection times, and transmission matrix. The latter is saved because it is stochastic
    # so we need to pass it to other functions to ensure consistency
    detection_time_file = os.path.join(save_folder, "detection_times.npy")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")
    np.save(detection_time_file, all_detection_times)
    np.save(clusters_file, np.array(clusters))
    np.save(beta_file, multi_beta)

def vaccination_strategy(tag, abridged_disease_params, patch_populations, sia_budget, unif_low=0.6, unif_high=1, sensitivities=0.03, 
                         num_days=1500, scale_factor=8.72*np.log(10),
                         chosen_patch=0, initial_cluster_allocation=1, operational_surveillance=None, detection_day_lag=0):
    
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

    # expand test sensitivities if needed
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = sensitivities*np.ones(num_patches)
    
    # if no site closures specified, assume all WES sites are operational
    if operational_surveillance is None:
        operational_surveillance = [True]*num_patches

    # combine preloaded beta with other disease parameters
    multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma = abridged_disease_params
    disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]
    
    # prepapre the initial state for the simulation
    pop_state = np.zeros(6*num_patches)
    pop_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
    frac_exposed = 0.01*pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[3*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state
    track_cumulative_totals = np.zeros(num_patches)
    
    # track which clusters have currently registered a detection event
    detections_in_cluster = [False]*n_clusters
    sia_intervention_allocated = False

    for _ in range(0, num_days):
        # simulate one day of disease evolution
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1,:]
        susceptible = pop_state[:num_patches]               # This is S1
        waned_susceptible = pop_state[num_patches:2*num_patches] # This is S2
        vaccinated = pop_state[2*num_patches:3*num_patches] # This is V
        exposures = pop_state[3*num_patches:4*num_patches]  # This is E
        infected = pop_state[4*num_patches:5*num_patches]

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

                # divide sia budget between patches, using allocation to decide how much goes to ground zero
                # vs other patches
                initial_cluster_budget = initial_cluster_allocation*sia_budget
                other_clusters_budget = (1-initial_cluster_allocation)*sia_budget/(n_clusters-1)
        
        # we have started the countdown until the detection is revealed and we act
        if sia_intervention_allocated:
            # this should only trigger once, implements delay from site sampling to detection announcement
            if countdown_to_vaccination == 0:
                # loop over all patches and provide allocated vaccination resources
                for j in range(0, num_patches):
                    if clusters[j] == ground_zero_cluster:
                        # additional_alpha = initial_cluster_budget/N[j]? or should we account for current demand and divide by susceptible?
                        multi_alpha[j] += initial_cluster_budget
                    else:
                        multi_alpha[j] += other_clusters_budget
            
            # do this at the end so that of this if statement so that a lag of zero actually means zero
            countdown_to_vaccination -= 1
    
    track_cumulative_totals = pop_state[4*num_patches:]
    return track_cumulative_totals, clusters

if __name__ == "__main__":
    tag = "total_population"
    run_through = True
    cluster_first = True

    # patch connection strengths
    in_patch = 10**(-2)
    out_patch = in_patch*10**(-1)

    # cluster information
    cluster_sizes = [50]*5  # 250 catchment sites in 5 subgroups
    num_patches = np.sum(cluster_sizes)
    n_clusters = len(cluster_sizes)

    # other disease parameters, given on day timescale
    gamma = np.log(1/0.9)/7
    delta = np.log(1/0.9)/10
    #delta = 0.01
    mu = 0.01
    c = 0.2
    alpha = 0.03
    beta = 0.3/1000
    omega = 0.001
    sigma= 0.001

    # for monitoring
    detection_threshold = 0.07
    close_site_frac = 0
    detect_lag=4

    one_patch_number = basic_reproduction_number(alpha, beta, mu, c, gamma, delta, N=1000)
    print(f"R_0 in isolated patch: {one_patch_number}")

    if run_through:
        # other disease parameters, given on day timescale
        multi_gamma = np.array([gamma]*num_patches)
        multi_delta = np.array([delta]*num_patches)
        multi_mu = np.array([mu]*num_patches)
        multi_c = np.array([c]*num_patches)
        multi_alpha = np.array([alpha]*num_patches)
        multi_sigma = np.array([sigma]*num_patches)
        multi_omega = np.array([omega]*num_patches)
        
        # construct a transmission matrix
        multi_beta = transmission_matrix(beta, cluster_sizes=cluster_sizes, in_cluster_strength=in_patch, out_cluster_strength=out_patch,
                                connect_to_frac=0.5)

        # package disease params, run simulations
        disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]
        abridged_disease_params = [multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]
        #sensitivities = 0.03*np.array([rng.uniform(low=0.95, high=2) for _ in range(0, num_patches)])
        sensitivities = detection_threshold*np.ones(num_patches)
        num_days = 600
        patch_pop = 1000*np.ones(num_patches)
        if cluster_first:
            clustering_scenario(tag, disease_params, patch_pop, cluster_sizes, unif_low=0.6, unif_high=1,
                                 sensitivities=sensitivities, num_days=num_days, detection_day_lag=detect_lag)

        # generate plots from clustering
        cwd = os.getcwd()
        save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
        clusters_file = os.path.join(save_folder, "clusters.npy")
        beta_file = os.path.join(save_folder, "beta.npy")
        detect_file = os.path.join(save_folder, "detection_times.npy")

        # load pre-existing results, redefines some variables from above
        clusters = np.load(clusters_file)
        all_detection_times = np.load(detect_file)
        n_clusters = len(set(clusters))  # how many non-duplicated cluster labels are present
        num_patches = len(clusters)
        multi_beta = np.load(beta_file)

        #cluster_accuracy(clusters, [50,50,50,50,50])
        #clustering_charts(all_detection_times, cluster_sizes, clusters)

        # how many sites are closed
        operational_surveillance = random_site_closure(clusters, close_site_frac)

        # right now, SIA budget is given in terms of alpha for simplicity. All patches have equal populations (1000), so in terms of daily doses
        # this is equal to 249*1000*alpha doses for each patch, which is probably a huge overcorrection
        sia_budget = (n_clusters-1)*alpha

        with_intervention, clusters = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop, 
                                                        sia_budget, sensitivities=sensitivities, num_days=num_days,
                                                            initial_cluster_allocation=1, operational_surveillance=operational_surveillance,
                                                            detection_day_lag=detect_lag)
        
        without_intervention, _ = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop, 
                                                    0, sensitivities=sensitivities, num_days=num_days, operational_surveillance=operational_surveillance,
                                                    detection_day_lag=detect_lag)
        
        spread_intervention, _ = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop,
                                                    sia_budget, initial_cluster_allocation=0, sensitivities=sensitivities, 
                                                    num_days=num_days, operational_surveillance=operational_surveillance, detection_day_lag=detect_lag)

        with_intervention_clustered = np.zeros(n_clusters+1)
        without_intervention_clustered = np.zeros(n_clusters+1)
        spread_clustered = np.zeros(n_clusters+1)
        for j in range(0, num_patches):
            with_intervention_clustered[clusters[j]] += with_intervention[j]
            without_intervention_clustered[clusters[j]] += without_intervention[j]
            spread_clustered[clusters[j]] += spread_intervention[j]

        with_intervention_clustered[-1] = np.mean(with_intervention_clustered[0:-1])
        without_intervention_clustered[-1] = np.mean(without_intervention_clustered[0:-1])
        spread_clustered[-1] = np.mean(spread_clustered[0:-1])

        # administrative variables for graphing
        originating_cluster = clusters[0]
        labels = [f"{j+1}" for j in range(0, n_clusters)]
        labels.append("Average")
        height = 1.25*max(without_intervention_clustered)
        bounding_factor = 1.23
        width=0.2

        plt.bar(np.arange(n_clusters+1)-width, without_intervention_clustered, width=width, label="No intervention")
        plt.bar(np.arange(n_clusters+1), with_intervention_clustered, width=width, label="Vaccinate origin cluster")
        plt.bar(np.arange(n_clusters+1)+width, spread_clustered, width=width, label="Vaccinate other clusters")
        plt.vlines(originating_cluster-2*width, 0, height/bounding_factor, color="black", linestyles="dashed")
        plt.vlines(originating_cluster+2*width, 0, height/bounding_factor, color="black", linestyles="dashed")
        plt.hlines(height/bounding_factor, originating_cluster-2*width, originating_cluster+2*width, color="black", linestyles="dashed", label="Origin cluster")
        plt.xticks(np.arange(n_clusters+1), labels)
        plt.ylim([0, height])
        plt.ylabel("Cumulative case count")
        plt.legend()
        plt.show()
    
