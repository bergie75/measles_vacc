# convenient file to define job to be done on the server
from train_tree import *
from parameters import exported_parameters
import os

#warnings.filterwarnings("ignore")
#thresholds = check_basic_reproduction_number(disease_params)
#print(f"Threshold 1: {thresholds[0]}, Threshold 2: {thresholds[1]}")
# reload = os.path.join("custom_starting_ensembles", "first_handcrafted")

number_of_members = exported_parameters["number_of_members"]
max_rounds = exported_parameters["max_rounds"]
optimize(number_of_members, max_rounds, 2, run_name="test_stochastic_update")