from trees import Node, DecisionTree, grow_random_tree, mate_trees
from state_simulator import score_tree
from parameters import *

import numpy as np

# create a simple ensemble of candidate trees
def generate_initial_candidates(number_of_members, depth=1):
    return [grow_random_tree(depth=depth) for _ in range(0, number_of_members)]

# yields the average score of a tree on num_simulations disease outbreaks, used to sort trees
def tree_training_score(candidate_tree):
    score = 0
    for _ in range(0, num_simulations):
        score += score_tree(candidate_tree)
    return score/num_simulations

def optimize(number_of_members, max_rounds, starting_depth):
    # source of randomness
    rng = np.random.default_rng()
    
    # initial candidate trees
    candidate_trees = generate_initial_candidates(number_of_members, depth=starting_depth)
    
    # main loop to upgrade our trees
    for i in range(0, max_rounds):
        #sort the trees based on how they score, we sort in ascending order since lower scores are
        #better
        print(f"Beginning tree fitness evaluation for round {i} ...")
        for tree_number, candidate in enumerate(candidate_trees):
            candidate.__setattr__("training_score", tree_training_score(candidate))
            if tree_number % 10 == 0:
                print(f"\tCompleted tree {tree_number}")
        print(f"All trees evaluated, sorting candidates and selecting top choices ...")
        sorted_candidates = sorted(candidate_trees, key=lambda x: x.training_score)
        candidate_trees = sorted_candidates[:top_choices]
        print(f"Candidates evaluated and top {top_choices} of {number_of_members} selected")
        
        print("Generating offspring from top candidates")
        for _ in range(0, number_of_members-top_choices):
            Parent1 = np.random.choice(sorted_candidates[:top_choices])
            Parent2 = np.random.choice(sorted_candidates[:top_choices])
            # recall that if trees can't be mated, the parents are returned as choices
            Child = np.random.choice(mate_trees(Parent1, Parent2))
            # double check the depth restriction here
            Child.mutate_tree(threshold_mutation_probability, decision_mutation_probability,
                               0.2, max_tree_depth-1)
            candidate_trees.append(Child)
        print("Offspring generated, round completed\n")

if __name__ == "__main__":
    optimize(number_of_members, max_rounds, 1)