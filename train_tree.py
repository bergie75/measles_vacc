from trees import Node, DecisionTree, grow_random_tree, mate_trees, load_tree
from state_simulator import score_tree
from parameters import *

import numpy as np
import warnings
import os
import shutil

# create a simple ensemble of candidate trees
def generate_initial_candidates(number_of_members, depth=1):
    return [grow_random_tree(depth=depth) for _ in range(0, number_of_members)]

def check_basic_reproduction_number(disease_params):
    beta, mu, c, gamma, delta = disease_params
    vax_rate = starting_vax_rate
    threshold_1 = mu*beta/(mu+vax_rate)*(c*mu+gamma+c*delta)/((mu+gamma)*(mu+delta))
    threshold_2 = c*beta*mu/((mu+vax_rate)*(2*mu+delta+gamma))
    return [threshold_1, threshold_2]

# yields the average score of a tree on num_simulations disease outbreaks, used to sort trees
def tree_training_score(candidate_tree):
    scores = []
    for _ in range(0, num_simulations):
        one_sim_score, _ = score_tree(candidate_tree)
        scores.append(one_sim_score)
    return np.mean(scores), np.std(scores)

def optimize(number_of_members, max_rounds, starting_depth=1,
              run_name="temp_experiment", reload=None, round=0):
    
    # create folder to save the run
    cwd = os.getcwd()
    run_home_folder = os.path.join(cwd, "measles_vacc", "run_data", run_name)
    if not os.path.exists(run_home_folder):
        os.makedirs(run_home_folder)

    # initial candidate trees
    if reload is None:
        candidate_trees = generate_initial_candidates(number_of_members, depth=starting_depth)
    else:
        candidate_trees = []
        load_from = os.path.join(cwd, "measles_vacc", reload, f"round_{round}")
        for file in os.listdir(load_from):
            if file.find(".pkl") != -1:
                reload_plus = os.path.join(reload, f"round_{round}")
                candidate_trees.append(load_tree(sub_folder=reload_plus, filename=file))
        while len(candidate_trees) < number_of_members:
            candidate_trees.append(grow_random_tree(depth=starting_depth))
    
    # use tree IDs to track the turnover between rounds
    old_ids = set([candidate.ID for candidate in candidate_trees])

    # place a copy of the parameters file as a text file in training folder to check parameters
    # used for training
    par_file = os.path.join(cwd, "measles_vacc", "parameters.py")
    text_file = os.path.join(run_home_folder, "archived_parameters.txt")
    shutil.copy(par_file, text_file)
    
    # open a logging file to store run information
    logfile = os.path.join(run_home_folder, "training_log.txt")

    with open(logfile, 'w') as log:
        # main loop to upgrade our trees
        for i in range(0, max_rounds):
            #sort the trees based on how they score, sort in ascending order since lower is better
            log.write(f"Beginning tree fitness evaluation for round {i} ...")
            print(f"Beginning tree fitness evaluation for round {i} ...")
            
            for candidate in candidate_trees:
                raw_score, score_std = tree_training_score(candidate)
                candidate.__setattr__("training_score", raw_score)
                candidate.__setattr__("score_std", score_std)
            
            log.write(f"All trees evaluated, sorting candidates and selecting top choices ...")
            print(f"All trees evaluated, sorting candidates and selecting top choices ...")
            sorted_candidates = sorted(candidate_trees, key=lambda x: x.training_score)
            candidate_trees = sorted_candidates[:top_choices]

            # quickly measure average fitness of top candidates for convergence and save trees
            fitness_check = 0
            spread = 0
            new_ids = []
            tree_depths = []
            round_folder = os.path.join(run_home_folder, f"round_{i}")
            os.makedirs(round_folder)

            for tree_index, curr_tree in enumerate(candidate_trees):
                fitness_check += curr_tree.training_score
                spread += curr_tree.score_std
                curr_tree.save(os.path.join(round_folder, f"tree_{tree_index}.pkl"))
                new_ids.append(curr_tree.ID)
                tree_depths.append(curr_tree.depth)
            
            # average the results over all trees
            fitness_check /= top_choices
            spread /= top_choices

            log.write(f"Average fitness of top {top_choices}/{number_of_members} members: {fitness_check}")
            print(f"Average fitness of top {top_choices}/{number_of_members} members: {fitness_check}")

            log.write(f"Average standard dev of training runs: {spread}")
            print(f"Average standard dev of training runs: {spread}")

            new_ids = set(new_ids)
            tree_persistence = 100*len(old_ids & new_ids)/len(new_ids)
            log.write(f"Percentage of trees persisting from last round: {tree_persistence:0.3f}%")
            print(f"Percentage of trees persisting from last round: {tree_persistence:0.3f}%")

            depth_mean = np.mean(tree_depths)
            depth_dev = np.std(tree_depths)
            log.write(f"Average depth: {depth_mean}, standard deviation: {depth_dev}")
            print(f"Average depth: {depth_mean}, standard deviation: {depth_dev}")
            
            log.write("Generating offspring from top candidates")
            print("Generating offspring from top candidates")
            
            for _ in range(0, number_of_members-top_choices):
                Parent1 = np.random.choice(sorted_candidates[:top_choices])
                Parent2 = np.random.choice(sorted_candidates[:top_choices])
                # recall that if trees can't be mated, the parents are returned as choices
                Child = np.random.choice(mate_trees(Parent1, Parent2))
                # double check the depth restriction here
                Child.mutate_tree(threshold_mutation_probability*threshold_attenuation**i,
                                decision_mutation_probability*decision_attenuation**i,
                                chop_decision_probability*chop_attenuation**i,
                                variable_change_probability*var_change_attenuation**i,
                                action_mutation_probability*action_attenuation**i,
                                0.2, max_tree_depth-1)
                candidate_trees.append(Child)
            
            log.write("Offspring generated, round completed\n")
            print("Offspring generated, round completed\n")

if __name__ == "__main__":
    #warnings.filterwarnings("ignore")
    #thresholds = check_basic_reproduction_number(disease_params)
    #print(f"Threshold 1: {thresholds[0]}, Threshold 2: {thresholds[1]}")
    pass