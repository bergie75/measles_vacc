from cluster_testing_Timas_copy import *
from random import sample
from numpy.linalg import eig
import random
from pathlib import Path
import numpy as np
import pandas as pd

random.seed(42) #not needed but added to compare with other code templates
 
rng = np.random.default_rng()

def vaccination_strategy_control_group(abridged_disease_params, patch_populations, extra_doses_per_day, multi_beta, num_days=600, chosen_patch=0):
    num_patches = len(patch_populations)
    patch_populations = np.array(patch_populations)

    # Unpack baseline disease parameters
    base_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma = abridged_disease_params
    base_alpha = np.copy(base_alpha) 

    # Prepare the initial state for the simulation
    pop_state = np.zeros(6 * num_patches)
    pop_state[:num_patches] = (multi_mu / (multi_mu + base_alpha)) * patch_populations
    
    # Seed the outbreak
    frac_exposed = 0.01 * pop_state[chosen_patch]
    pop_state[chosen_patch] -= frac_exposed
    pop_state[3 * num_patches + chosen_patch] += frac_exposed

    # Main daily epidemic simulation loop
    for day in range(0, num_days):
        # Reset to baseline every day
        multi_alpha = np.copy(base_alpha)
        
        # Deploy your maximum extra vaccine budget aggressively during the first week
        if day < 7:
            doses_per_patch = extra_doses_per_day / num_patches
            for j in range(0, num_patches):
                S1 = pop_state[j]
                if S1 > 1:
                    additional_alpha = doses_per_patch / S1
                    multi_alpha[j] += additional_alpha

        # Package active parameters
        disease_params = [multi_alpha, multi_beta, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]
        
        # Advance simulation by 1 day
        pop_state = odeint(compartment_rhs_multi_patch, pop_state, [0, 1], args=(disease_params, num_patches, patch_populations))[-1, :]
         
    # Extract total cumulative cases at the very end
    track_cumulative_totals = pop_state[5 * num_patches : 6 * num_patches]    
    return track_cumulative_totals

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    
    # Structural network configuration
    in_patch = 10**(-2)
    out_patch = in_patch * 10**(-1)
    cluster_sizes = [50] * 5  
    num_patches = np.sum(cluster_sizes)

    # Base disease parameters
    gamma = np.log(1/0.9)/7
    delta = np.log(1/0.9)/10
    mu = 0.01
    c = 0.2
    alpha = 0.03
    beta = 0.15 / 1000 
    omega = 1
    sigma = 1
    num_days = 600
    initial_exposure_patch = 0
    
    # Conversion metrics
    site_to_dose_conversion = 70.29

    # Generate arrays
    multi_gamma = np.array([gamma] * num_patches)
    multi_delta = np.array([delta] * num_patches)
    multi_mu = np.array([mu] * num_patches)
    multi_c = np.array([c] * num_patches)
    multi_alpha = np.array([alpha] * num_patches)
    multi_sigma = np.array([sigma] * num_patches)
    multi_omega = np.array([omega] * num_patches)
    
    # Construct transmission matrix directly
    multi_beta = transmission_matrix(beta, cluster_sizes=cluster_sizes, in_cluster_strength=in_patch, out_cluster_strength=out_patch, connect_to_frac=0.5)
    abridged_disease_params = [multi_alpha, multi_mu, multi_c, multi_gamma, multi_delta, multi_omega, multi_sigma]
    
    # Population configuration
    big_pops = [5000, 5000, 5000, 5000, 5000]
    big_pops.extend([550] * 45)
    patch_pop = np.array(big_pops * 5)

    # ALL SITES ARE CLOSED
    extra_doses = num_patches * site_to_dose_conversion
    print(f"Total maximum daily dose budget allocated: {extra_doses} doses/day")

    num_samples = 250
    total_control_case_counts = np.zeros(num_patches)

    print("Starting simulation loops for fully closed surveillance control group")
    for i in range(0, num_samples):
        # Call the simplified function structure
        cumulative_case_counts_sample = vaccination_strategy_control_group(
            deepcopy(abridged_disease_params), patch_pop, extra_doses, multi_beta,
            num_days=num_days, chosen_patch=initial_exposure_patch
        )
        
        total_control_case_counts += cumulative_case_counts_sample

    # Compute true sample average across the runs
    total_control_case_counts /= num_samples
    
    # Calculate the overall global aggregate case load across all 250 patches
    total_infected_population = np.sum(total_control_case_counts)

    print(f"--- Control Group Simulation Complete ---")
    print(f"Total number of people infected by Day {num_days}: {total_infected_population:.2f}")
    
    # Save your results matrix cleanly
    save_folder = script_dir / "sigma=1, omega=1 data control_group"
    save_folder.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(str(save_folder), "control_total_cases.npy"), total_control_case_counts)