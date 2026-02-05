import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from dtw import dtw
import os
import time

rng = np.random.default_rng()

def create_wes_data(exposure_timeseries, scale_factor=1000*8.72*np.log(10), std_frac=10**(-2), minimum=0):
    perfect_data = exposure_timeseries*scale_factor

    measurement_outputs = []
    for result in perfect_data:
        if result > minimum:
            measurement_mean = np.log(result)
            measurement_std = np.abs(measurement_mean)*std_frac
            measurement_outputs.append(np.exp(rng.normal(measurement_mean, measurement_std)))
        else:
            measurement_outputs.append(0)
    
    return np.array(measurement_outputs)

def random_upper_triangular(num_patches, strength=1):
    A = np.zeros((num_patches, num_patches))
    for i in range(0, num_patches):
        for j in range(i+1, num_patches):
            A[i,j] = strength*rng.uniform()
    
    return A

def transmission_matrix(raw_beta, cluster_sizes,
                         in_cluster_strength=0.5, out_cluster_strength=0.01):
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
                upper_triangle[i,j] = in_cluster_strength*rng.uniform()
            # out of cluster entries that come after the block diagonal
            for j in range(starting_index+cluster_size, num_patches):
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

def dtw_test_run():
    num_patches = 200
    initial_state = [1]*num_patches

    # other disease parameters, given on day timescale
    gamma = np.array([np.log(2)/3]*num_patches)
    delta = np.array([np.log(2)/3]*num_patches)
    mu = np.array([0.001]*num_patches)
    c = np.array([1]*num_patches)
    alpha = np.array([0.03]*num_patches)
    
    # construct a transmission matrix
    raw_beta = 0.014
    beta = transmission_matrix(raw_beta, cluster_sizes=[100,100], in_cluster_strength=1, out_cluster_strength=10**(-2))

    # package for convenience
    chosen_patch = 0
    disease_params = [alpha, beta, mu, c, gamma, delta]
    initial_state.extend([0]*3*num_patches)
    initial_state = initial_state
    initial_state[chosen_patch] -= 0.01
    initial_state[2*num_patches+chosen_patch] += 0.01
    
    timepoints = np.linspace(0,50,10000)
    disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches))
    
    # break out solutions for one patch
    susceptible = disease_sim[:,0]
    vaccinations = disease_sim[:,num_patches]
    exposed = disease_sim[:,2*num_patches]
    infections = disease_sim[:,3*num_patches]
    resistant = 1-susceptible-vaccinations-exposed-infections

    exposures = disease_sim[:,2*num_patches:3*num_patches]
    for i in range(0, num_patches):
        plt.plot(timepoints, exposures[:,i])
    plt.show()

    start_time = time.time()
    dtw_distances=np.zeros((num_patches, num_patches))
    for i in range(0, num_patches):
        if i % 10 == 0:
            print(f"Completed up to row {i}")
        for j in range(i+1, num_patches):
            dtw_distances[i,j] = dtw(exposures[:,i], exposures[:,j]).distance

    end_time = time.time()
    print(f"Operation took {(end_time-start_time)/3600} hours")
    
    # save computed distances
    cwd = os.getcwd()
    clustering_folder = os.path.join(cwd, "clustering_data")
    save_loc = os.path.join(clustering_folder, "cluster_distances.npy")
    np.save(save_loc, dtw_distances)

if __name__ == "__main__":
    num_patches = 200
    initial_state = [1]*num_patches

    # other disease parameters, given on day timescale
    gamma = np.array([np.log(2)/3]*num_patches)
    delta = np.array([np.log(2)/3]*num_patches)
    mu = np.array([0.001]*num_patches)
    c = np.array([1]*num_patches)
    alpha = np.array([0.03]*num_patches)
    
    # construct a transmission matrix
    raw_beta = 0.014
    beta = transmission_matrix(raw_beta, cluster_sizes=[100,100], in_cluster_strength=1, out_cluster_strength=10**(-2))

    # package for convenience
    chosen_patch = 0
    disease_params = [alpha, beta, mu, c, gamma, delta]
    initial_state.extend([0]*3*num_patches)
    initial_state = initial_state
    initial_state[chosen_patch] -= 0.01
    initial_state[2*num_patches+chosen_patch] += 0.01
    
    timepoints = np.linspace(0,50,10000)
    disease_sim = odeint(compartment_rhs_multi_patch, initial_state, timepoints, args=(disease_params, num_patches))
    
    # break out solutions for one patch
    susceptible = disease_sim[:,0]
    vaccinations = disease_sim[:,num_patches]
    exposed = disease_sim[:,2*num_patches]
    infections = disease_sim[:,3*num_patches]
    resistant = 1-susceptible-vaccinations-exposed-infections

    exposures = disease_sim[:,2*num_patches:3*num_patches]
    for i in range(0, num_patches):
        plt.plot(timepoints, exposures[:,i])
    plt.show()

    start_time = time.time()
    dtw_distances=np.zeros((num_patches, num_patches))
    for i in range(0, num_patches):
        if i % 10 == 0:
            print(f"Completed up to row {i}")
        for j in range(i+1, num_patches):
            dtw_distances[i,j] = dtw(exposures[:,i], exposures[:,j]).distance

    end_time = time.time()
    print(f"Operation took {(end_time-start_time)/3600} hours")
    
    # save computed distances
    cwd = os.getcwd()
    clustering_folder = os.path.join(cwd, "clustering_data")
    save_loc = os.path.join(clustering_folder, "cluster_distances.npy")
    np.save(save_loc, dtw_distances)