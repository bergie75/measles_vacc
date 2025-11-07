# convenient file to define job to be done on the server
from train_tree import *
import os

#warnings.filterwarnings("ignore")
#thresholds = check_basic_reproduction_number(disease_params)
#print(f"Threshold 1: {thresholds[0]}, Threshold 2: {thresholds[1]}")
reload = os.path.join("custom_starting_ensembles", "first_handcrafted")
optimize(number_of_members, max_rounds, 2, reload=reload, run_name="less_infection_cost_less_response")