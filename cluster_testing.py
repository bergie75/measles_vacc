import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as colormap
from dtw import dtw
import os
import time
from copy import deepcopy

rng = np.random.default_rng()

def make_true_cluster_vec(cluster_sizes):
    true_cluster_vec = []
    for i, cluster_size in enumerate(cluster_sizes):
        true_cluster_vec.extend([i]*cluster_size)
    
    return true_cluster_vec

def find_detection_times(wes_data, timepoints, sensitivities, scale_factor=8.72*np.log(10), N=None):
    if N is None:
        num_patches = len(sensitivities)
        N=np.ones(num_patches)
    
    # perform stacking for compatibility with rpevious code
    num_timepoints = len(timepoints)
    sensitivities = np.vstack([sensitivities*N]*num_timepoints)
    
    # a vector to hold the detection times
    detection_times = []

    # precursor to detection times
    detections = (wes_data >= (sensitivities*scale_factor))

    for j in range(0, wes_data.shape[1]):
        for i, time in enumerate(timepoints):
            if detections[i,j]:
                detection_times.append(time)
                break
    
    return np.array(detection_times)

def basic_reproduction_number(alpha, beta, mu, c, gamma, delta, N=1):
    return N*beta*mu/(mu+alpha)*(c*(mu+delta)+gamma)/((mu+gamma)*(mu+delta))

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

def compartment_rhs_multi_patch(x, t, disease_params, num_patches, N):
    # if population sizes are not specified, assume fractions
    if N is None:
        N=np.ones(num_patches)
    
    # break state up into epidemiologically relevant categories
    S=x[:num_patches]
    V=x[num_patches:2*num_patches]
    E=x[2*num_patches:3*num_patches]
    I=x[3*num_patches:4*num_patches]
    #cum_cases = x[4*num_patches:]

    alpha, beta, mu, c, gamma, delta = disease_params

    dS_dt = mu*(N-S)-alpha*S-np.matmul(beta,(c*E+I))*S
    dV_dt = alpha*S-mu*V
    dE_dt = np.matmul(beta,(c*E+I))*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I
    dcum_cases_dt = gamma*E-(mu+delta)*I

    return np.concatenate((dS_dt, dV_dt, dE_dt, dI_dt, dcum_cases_dt))

def cluster_accuracy(clusters, cluster_sizes):
    # this should probably just be called a confusion matrix. Measures how close clusters are to underlying
    # construction of transmission matrix
    n_clusters = len(set(clusters))
    true_clusters = np.array(make_true_cluster_vec(cluster_sizes))

    commonality_matrix = np.zeros((n_clusters, n_clusters))
    for i,pred_cluster in enumerate(clusters):
        commonality_matrix[pred_cluster, true_clusters[i]] += 1
    
    # attempt to put dominant terms on diagonal
    permutation = np.zeros((n_clusters, n_clusters))
    for i in range(0, n_clusters):
        j=np.argmax(commonality_matrix[i,:])
        permutation[j,i]=1
    
    # check for collisions and abort attempt to reorganize matrix if these occur
    valid_permutation = False
    for j in range(0, n_clusters):
        col_check = (np.sum(permutation[:,j]) == 1)
        valid_permutation = (valid_permutation or col_check)
    
    if valid_permutation:
        commonality_matrix = np.matmul(commonality_matrix, permutation)
    
    sns.heatmap(commonality_matrix, xticklabels=False, yticklabels=[f"Cluster {j+1}" for j in range(0, n_clusters)])
    plt.show()

def clustering_charts(all_detection_times, cluster_sizes, clusters, cluster_cutoff=1):
        # plot detection times within and between clusters
    lower_index = 0
    n_clusters = len(cluster_sizes)
    num_patches = np.sum(cluster_sizes)
    
    # don't alwats show every possibility
    if cluster_cutoff is None:
        display_clusters = cluster_sizes
    else:
        display_clusters = cluster_sizes[:cluster_cutoff]
    for cluster_size in display_clusters:
        final_cluster_detections = [[] for _ in range(0, n_clusters)]
        upper_index = lower_index + cluster_size
        detections_start_in_cluster = all_detection_times[lower_index:upper_index,:]
        for j in range(0, num_patches):
            final_cluster_detections[clusters[j]].extend(detections_start_in_cluster[:,j])
        
        final_averages = [np.mean(x) for x in final_cluster_detections]
        final_std = [np.std(x) for x in final_cluster_detections]
        
        # update for next cluster
        lower_index = upper_index
        height = max(final_averages+final_std)*1.1
        labels = [f"Cluster {j+1}" for j in range(0, n_clusters)]
        plt.bar([j+1 for j in range(0, n_clusters)], final_averages, label="Average detection time")
        plt.errorbar([j+1 for j in range(0, n_clusters)], final_averages, yerr=final_std, capsize=6, color="black", linestyle='', label="Standard dev. of detection time")
        plt.xticks(1+np.arange(n_clusters), labels)
        plt.ylabel("Detection time (days)")
        plt.ylim([0, height])
        plt.legend()
        plt.show()

def clustering_scenario(tag, disease_params, patch_populations, cluster_sizes, unif_low=0.6, unif_high=1, sensitivities=0.03, num_days=1500,
                        minimum=0, scale_factor=8.72*np.log(10)):
    # unpack useful variables
    multi_alpha, multi_beta, multi_mu, _, _, _ = disease_params
    num_patches = np.sum(cluster_sizes)
    num_timepoints=num_days+1
    timepoints = np.linspace(0,num_days,num_timepoints)
    all_detection_times = np.zeros((num_patches, num_patches))
    n_clusters = len(cluster_sizes)
    
    # expand test sensitivities. If scalar given, all sensitivities are the same. If vector, then sensitivity varies by catchment
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = sensitivities*np.ones(num_patches)
    
    # start all catchment areas at disease free equilibrium vaccination level
    for chosen_patch in range(0, num_patches):
        print(f"Computing patch {chosen_patch}")
        initial_state = np.zeros(5*num_patches)
        initial_state[:num_patches] = (multi_mu/(multi_mu+multi_alpha))*patch_populations
        initial_state[num_patches:2*num_patches] = (multi_alpha/(multi_mu+multi_alpha))*patch_populations
        frac_exposed = 0.01*initial_state[chosen_patch]
        initial_state[chosen_patch] -= frac_exposed
        initial_state[2*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state

        disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches, patch_populations))
        exposures = disease_sim[:,2*num_patches:3*num_patches]
        infected = disease_sim[:,3*num_patches:4*num_patches]

        # save results of wastewater
        wes_data = np.zeros(exposures.shape)
        for i in range(0, num_patches):
            wes_data[:,i] = create_wes_data(exposures[:,i]+infected[:,i], unif_low=unif_low, unif_high=unif_high, minimum=minimum,
                                             scale_factor=scale_factor)
        
        # compute when disease is actually found in wastewater
        all_detection_times[chosen_patch,:] = find_detection_times(wes_data, timepoints, sensitivities, 
                                                                   scale_factor=scale_factor, N=patch_populations)
    
    # from wastewater data alone, attempt to cluster the catchment areas
    agg = KMeans(n_clusters=n_clusters)
    learned_clusters = np.array(agg.fit_predict(all_detection_times))

    cluster_accuracy(learned_clusters, cluster_sizes)

    # plot detection times within and between clusters
    clustering_charts(all_detection_times, cluster_sizes, learned_clusters)
    
    # create save directory
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # save found clusters, detection times, and transmission matrix. The latter is saved because it is stochastic
    # so we need to pass it to other functions to ensure consistency
    detection_time_file = os.path.join(save_folder, "detection_times.npy")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")
    np.save(detection_time_file, all_detection_times)
    np.save(clusters_file, np.array(learned_clusters))
    np.save(beta_file, multi_beta)

def vaccination_strategy(tag, abridged_disease_params, patch_populations, sia_budget, unif_low=0.6, unif_high=1, sensitivities=0.03, 
                         num_days=1500, scale_factor=8.72*np.log(10),
                         chosen_patch=0, initial_cluster_allocation=1, operational_surveillance=None):
    
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
    sia_intervention_used = False

    for _ in range(0, num_days):
        # simulate one day of disease evolution
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1,:]
        exposures = pop_state[2*num_patches:3*num_patches]
        infected = pop_state[3*num_patches:4*num_patches]

        # we get one sia per simulation, check if it has been used
        if not sia_intervention_used:
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
                sia_intervention_used = True
                # most of the time this is picking from a list of length one?
                ground_zero_cluster = np.random.choice(cluster_indices_with_detections)

                # divide sia budget between patches, using allocation to decide how much goes to ground zero
                # vs other patches
                initial_cluster_budget = initial_cluster_allocation*sia_budget
                other_clusters_budget = (1-initial_cluster_allocation)*sia_budget/(n_clusters-1)
                
                # loop over all patches and provide allocated vaccination resources
                for j in range(0, num_patches):
                    if clusters[j] == ground_zero_cluster:
                        multi_alpha[j] += initial_cluster_budget
                    else:
                        multi_alpha[j] += other_clusters_budget
    
    track_cumulative_totals = pop_state[4*num_patches:]
    return track_cumulative_totals, clusters

if __name__ == "__main__":
    tag = "alternative_parameters"
    run_through = True
    cluster_first = False

    # patch connection strengths
    in_patch = 10**(-1)
    out_patch = in_patch*10**(-2)

    # cluster information
    cluster_sizes = [50,50,50,50,50]  # 250 catchment sites in 5 subgroups
    num_patches = np.sum(cluster_sizes)
    n_clusters = len(cluster_sizes)

    # other disease parameters, given on day timescale
    gamma = np.log(1/0.9)/7
    delta = np.log(1/0.9)/10
    #delta = 0.01
    mu = 0.01
    c = 0.2
    alpha = 0.05
    beta = 0.3/1000

    # for monitoring
    detection_threshold = 0.07

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
        #sensitivities = 0.03*np.array([rng.uniform(low=0.95, high=2) for _ in range(0, num_patches)])
        sensitivities = detection_threshold*np.ones(num_patches)
        num_days = 600
        patch_pop = 1000*np.ones(num_patches)
        if cluster_first:
            clustering_scenario(tag, disease_params, patch_pop, cluster_sizes, unif_low=0.6, unif_high=1, sensitivities=sensitivities, num_days=num_days)

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
        operational_surveillance = random_site_closure(clusters, 0.99)

        sia_budget = (n_clusters-1)*alpha
        with_intervention, clusters = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop, 
                                                        sia_budget, sensitivities=sensitivities, num_days=num_days,
                                                            initial_cluster_allocation=1, operational_surveillance=operational_surveillance)
        
        without_intervention, _ = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop, 
                                                    0, sensitivities=sensitivities, num_days=num_days, operational_surveillance=operational_surveillance)
        
        spread_intervention, _ = vaccination_strategy(tag, deepcopy(abridged_disease_params), patch_pop,
                                                    sia_budget, initial_cluster_allocation=0, sensitivities=sensitivities, 
                                                    num_days=num_days, operational_surveillance=operational_surveillance)

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
        labels = [f"Cluster {j+1}" for j in range(0, n_clusters)]
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
    
