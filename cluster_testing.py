import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from dtw import dtw
import os
import time

rng = np.random.default_rng()

def make_true_cluster_vec(cluster_sizes):
    true_cluster_vec = []
    for i, cluster_size in enumerate(cluster_sizes):
        true_cluster_vec.extend([i]*cluster_size)
    
    return true_cluster_vec

def find_detection_times(wes_data, timepoints, sensitivities, scale_factor=1000*8.72*np.log(10)):
    # a vector to hold the detection times
    detection_times = []

    # precursor to detection times
    detections = (wes_data >= sensitivities*scale_factor)

    for j in range(0, wes_data.shape[1]):
        for i, time in enumerate(timepoints):
            if detections[i,j]:
                detection_times.append(time)
                break
    
    return np.array(detection_times)

def basic_reproduction_number(alpha, beta, mu, c, gamma, delta):
    return beta*mu/(mu+alpha)*(c*(mu+delta)+gamma)/((mu+gamma)*(mu+delta))

def create_wes_data(exposure_timeseries, scale_factor=1000*8.72*np.log(10), unif_low=1, unif_high=1, minimum=0):
    perfect_data = exposure_timeseries*scale_factor

    measurement_outputs = []
    for result in perfect_data:
        messy_result = result*rng.uniform(low=unif_low, high=unif_high)
        if messy_result > minimum:
            measurement_outputs.append(messy_result)
        else:
            measurement_outputs.append(0)
    
    return np.array(measurement_outputs)

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

def compartment_rhs_multi_patch(x, t, disease_params, num_patches):
    # break state up into epidemiologically relevant categories
    S=x[:num_patches]
    V=x[num_patches:2*num_patches]
    E=x[2*num_patches:3*num_patches]
    I=x[3*num_patches:]

    alpha, beta, mu, c, gamma, delta = disease_params

    dS_dt = mu*(1-S)-alpha*S-np.matmul(beta,(c*E+I))*S
    dV_dt = alpha*S-mu*V
    dE_dt = np.matmul(beta,(c*E+I))*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.concatenate((dS_dt, dV_dt, dE_dt, dI_dt))

def dtw_test_run(disease_params, cluster_sizes, num_timepoints=1000, std_frac=10**(-2), minimum=0,
                  tag="default", show_plots=True):
    num_patches = np.sum(cluster_sizes)
    initial_state = [1]*num_patches

    alpha, beta, mu, c, gamma, delta = disease_params

    # package for convenience
    chosen_patch = 0
    disease_params = [alpha, beta, mu, c, gamma, delta]
    initial_state.extend([0]*3*num_patches)
    initial_state = initial_state
    initial_state[chosen_patch] -= 0.01
    initial_state[2*num_patches+chosen_patch] += 0.01
    
    timepoints = np.linspace(0,50,num_timepoints)
    disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches))
    
    # break out solutions for one patch
    # susceptible = disease_sim[:,0]
    # vaccinations = disease_sim[:,num_patches]
    # exposed = disease_sim[:,2*num_patches]
    # infections = disease_sim[:,3*num_patches]
    # resistant = 1-susceptible-vaccinations-exposed-infections
    
    # find data on exposures
    exposures = disease_sim[:,2*num_patches:3*num_patches]
    if show_plots:
        for i in range(0, num_patches):
            check_val = i
            for k,elem in enumerate(cluster_sizes):
                check_val -= elem
                if check_val <=0:
                    cluster = k
                    break
            plt.plot(timepoints, exposures[:,i], color=colormap.hot(cluster/(len(cluster_sizes)-1)))
        plt.show()

    # save results of wastewater
    wes_data = np.zeros(exposures.shape)
    for i in range(0, num_patches):
        wes_data[:,i] = create_wes_data(exposures[:,i], std_frac=std_frac, minimum=minimum)

    start_time = time.time()
    dtw_distances=np.zeros((num_patches, num_patches))
    for i in range(0, num_patches):
        if i % 10 == 0:
            print(f"Completed up to row {i}")
        for j in range(i+1, num_patches):
            # compute distance between wes measurements
            elem_i=np.cumsum(wes_data[i,:])
            elem_j=np.cumsum(wes_data[j,:])
            dtw_distances[i,j] = dtw(elem_i, elem_j).distance

    end_time = time.time()
    print(f"Operation took {(end_time-start_time)/3600} hours")
    
    # save computed distances
    cwd = os.getcwd()
    clustering_folder = os.path.join(cwd, "clustering_data")
    cluster_distances_file = os.path.join(clustering_folder, f"cluster_distances_{tag}.npy")
    raw_trajectories_file = os.path.join(clustering_folder, f"trajectories_{tag}.npy")
    np.save(cluster_distances_file, dtw_distances)
    np.save(raw_trajectories_file, wes_data)

def measure_clustering(tag="default", n_clusters=2):
    # create file names to load from
    cwd = os.getcwd()
    clustering_folder = os.path.join(cwd, "clustering_data")
    cluster_distances_file = os.path.join(clustering_folder, f"cluster_distances_{tag}.npy")

    # load results
    computed_distances = np.load(cluster_distances_file)
    raw_trajectories_file = os.path.join(clustering_folder, f"trajectories_{tag}.npy")
    distance_matrix = computed_distances + np.transpose(computed_distances)
    raw_trajectories = np.load(raw_trajectories_file)

    # compute detection times
    detection_times = find_detection_times(raw_trajectories, np.linspace(0,50,1000), 0.03*np.ones(raw_trajectories.shape))

    agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    clusters = agg.fit_predict(distance_matrix)

    for i in range(0, raw_trajectories.shape[1]):
        plt.plot(raw_trajectories[:,i], color=colormap.hot(clusters[i]/(n_clusters-1)))
    plt.show()

    return clusters

def detection_clustering(disease_params, cluster_sizes, num_days=1000, unif_low=1, unif_high=1, minimum=0,
                  sensitivity=0.1, scale_factor=8.72*1000*np.log(10)):
    num_patches = np.sum(cluster_sizes)
    num_timepoints=num_days+1
    timepoints = np.linspace(0,num_days,num_timepoints)
    sensitivities = sensitivity*np.ones((num_timepoints, num_patches))
    all_detection_times = np.zeros((num_patches, num_patches))
    n_clusters = len(cluster_sizes)

    # for initial vaccination
    alpha, _, mu, _, _, _ = disease_params

    for chosen_patch in range(0, num_patches):
        print(f"Computing patch {chosen_patch}")
        initial_state = np.zeros(4*num_patches)
        initial_state[:num_patches] = mu/(mu+alpha)
        initial_state[num_patches:2*num_patches] = alpha/(mu+alpha)
        frac_exposed = 0.01*initial_state[chosen_patch]
        initial_state[chosen_patch] -= frac_exposed
        initial_state[2*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state

        disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches))
        exposures = disease_sim[:,2*num_patches:3*num_patches]
        infected = disease_sim[:,3*num_patches:4*num_patches]

        # save results of wastewater
        wes_data = np.zeros(exposures.shape)
        for i in range(0, num_patches):
            wes_data[:,i] = create_wes_data(exposures[:,i]+infected[:,i], unif_low=unif_low, unif_high=unif_high, minimum=minimum, scale_factor=scale_factor)
        
        # compute when disease is actually found in wastewater
        all_detection_times[chosen_patch,:] = find_detection_times(wes_data, timepoints, sensitivities, scale_factor=scale_factor)
    
    agg = KMeans(n_clusters=n_clusters)
    clusters = np.array(agg.fit_predict(all_detection_times))
    true_clusters = np.array(make_true_cluster_vec(cluster_sizes))

    commonality_matrix = np.zeros((n_clusters, n_clusters))
    for i,pred_cluster in enumerate(clusters):
        commonality_matrix[pred_cluster, true_clusters[i]] += 1
    
    permutation = np.zeros((n_clusters, n_clusters))
    for i in range(0, n_clusters):
        j=np.argmax(commonality_matrix[i,:])
        permutation[j,i]=1
    
    valid_permutation = False
    for j in range(0, n_clusters):
        col_check = (np.sum(permutation[:,j]) == 1)
        valid_permutation = (valid_permutation or col_check)
    
    if valid_permutation:
        commonality_matrix = np.matmul(commonality_matrix, permutation)
    
    print(commonality_matrix)

    return clusters, all_detection_times

def clustering_scenario(tag, disease_params, cluster_sizes, unif_low=0.6, unif_high=1, sensitivity=0.03, num_days=1500):
    _, multi_beta, _, _, _, _ = disease_params
    clusters, all_detection_times = detection_clustering(disease_params, cluster_sizes, unif_low=unif_low, unif_high=unif_high, sensitivity=sensitivity, num_days=num_days)

    lower_index = 0
    for cluster_size in cluster_sizes:
        final_cluster_detections = [[] for _ in range(0, n_clusters)]
        upper_index = lower_index + cluster_size
        detections_start_in_cluster = all_detection_times[lower_index:upper_index,:]
        for j in range(0, num_patches):
            final_cluster_detections[clusters[j]].extend(detections_start_in_cluster[:,j])
        
        final_averages = [np.mean(x) for x in final_cluster_detections]
        final_std = [np.std(x) for x in final_cluster_detections]
        
        # update for next cluster
        lower_index = upper_index
        plt.bar([1,2,3,4,5], final_averages, yerr=final_std)
        plt.show()
    
    # save results
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    detection_time_file = os.path.join(save_folder, "detection_times.npy")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")
    np.save(detection_time_file, all_detection_times)
    np.save(clusters_file, np.array(clusters))
    np.save(beta_file, multi_beta)

def vaccination_strategy(tag, abridged_disease_params, unif_low=0.6, unif_high=1, sensitivity=0.03, num_days=1500):
    # load data from clustering analysis
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "clustering_data", f"{tag}")
    clusters_file = os.path.join(save_folder, "clusters.npy")
    beta_file = os.path.join(save_folder, "beta.npy")

    clusters = np.load(clusters_file)
    num_patches = len(clusters)
    multi_beta = np.load(beta_file)
    multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta = abridged_disease_params
    disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta]

    chosen_patch = 30  # we will make this user selectable later
    
    # prepapre the initial state for the simulation
    pop_state = np.zeros(4*num_patches)
    pop_state[:num_patches] = mu/(mu+alpha)
    pop_state[num_patches:2*num_patches] = alpha/(mu+alpha)
    frac_exposed = 0.01*pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[2*num_patches+chosen_patch] += frac_exposed  # place patients in exposed state

    for day in range(0, num_days):
    # simulate one day of disease evolution
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches))[-1,:]
        exposures = pop_state[2*num_patches:3*num_patches]
        infected = pop_state[3*num_patches:4*num_patches]


if __name__ == "__main__":
    tag = "test_script"

    # patch connection strengths
    in_patch = 3*10**(-2)
    out_patch = in_patch*10**(-3)

    # cluster information
    cluster_sizes = [50,50,50,50,50]  # 250 catchment sites in 5 subgroups
    num_patches = np.sum(cluster_sizes)
    n_clusters = len(cluster_sizes)

    # other disease parameters, given on day timescale
    gamma = np.log(2)/7
    delta = np.log(2)/7
    mu = 0.01
    c = 0
    alpha = 0.02
    beta = 0.6

    one_patch_number = basic_reproduction_number(alpha, beta, mu, c, gamma, delta)
    print(f"R_0 in isolated patch: {one_patch_number}")
    many_patch_estimate = (1+(cluster_sizes[0]-1)/2*in_patch + cluster_sizes[1]/2*out_patch)*one_patch_number
    print(f"Overall R_0: {many_patch_estimate}")

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
    clustering_scenario(tag, disease_params, cluster_sizes, unif_low=0.6, unif_high=1, sensitivity=0.03, num_days=1500)
    
