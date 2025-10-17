import numpy as np

num_simulations = 1000
max_simulation_depth = 180
max_pop = 5000

action_set = ["increase_vax_rate", "decrease_vax_rate", "apply_npi", "remove_npi", "diagnostic_measurement", "wes_measurement"]
input_threshold_ranges = {"time_since_diag": [0, max_simulation_depth],
                    "time_since_wes": [0, max_simulation_depth],
                      "current_diag": [0, max_pop],
                        "current_wes": [0, max_pop]}

# describes the overall range of choices on vaccination
default_vax_rate = 0.01
vax_rate_modifier = 1.05

npi_modifier = 0.9

# affect measurements
wes_std_frac = 0.1  # determines how noisy wastewater surveillance is
infected_seeking_care_frac = 0.2

# disease dynamics parameters
disease_params = np.array([1,1,1,1])