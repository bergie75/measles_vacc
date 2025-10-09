import numpy as np
import random

max_simulation_depth = 180
max_pop = 5000
action_set = ["increase_vax_rate", "apply_npi", "diagnostic_measurement", "wes_measurement"]
input_threshold_ranges = {"time_since_diag": [0, max_simulation_depth],
                    "time_since_wes": [0, max_simulation_depth],
                      "current_diag": [0, max_pop],
                        "current_wes": [0, max_pop]}

class Node:
    def __init__(self, action=None, decision_var=None, threshold=None, left_child=None, right_child=None):
        self.action = action
        self.decision_var = decision_var
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

    def __repr__(self):
        if self.action is not None:
            return self.action
        elif (self.threshold is not None) and (self.decision_var is not None):
            return f"{self.decision_var} >= {self.threshold}"
        else:
            return "ERROR: malformed node"

    def terminal_leaf(self):
        return (self.action is not None)
    
    # defaults to taking a new value between 80% and 120% of old value
    # enforces threshold limits
    def mutate_threshold(self, max_perc_change=0.2):
        rng = np.random.default_rng()
        multiplier = rng.uniform(low=1-max_perc_change, high=1+max_perc_change)
        candidate_value = self.threshold*multiplier

        if input_threshold_ranges[self.decision_var][0] <= candidate_value <= input_threshold_ranges[self.decision_var][0]:
            self.threshold = candidate_value
    
    # inputs will be in the form of a dictionary {str(name_decision_var): float(value_of_the_variable)}
    def evaluate(self, inputs):
        # terminal condition to end the recursion
        if self.terminal_leaf():
            return self.action
        
        # check threshold to determine the branch to follow, uses dictionary structure of inputs
        if inputs[self.decision_var] <= self.threshold:
            return self.left_child.evaluate(inputs)
        else:
            return self.right_child.evaluate(inputs)
    
    def get_depth(self):
        # base case of recursion
        if self.action is not None:
            return 1
        
        return 1 + max(self.left_child.get_depth(), self.right_child.get_depth())


class DecisionTree:
    def __init__(self, root_node=None):
        self.root_node = root_node

        # keep track of how deep the tree extends
        if root_node is None:
            self.depth = -1
        else:
            self.depth = self.root_node.get_depth()

        # set up a dictionary that describes the nodes at each level of the tree for easy reference
        node_dictionary = {0: [self.root_node]}
        for i in range(1, self.depth):
            node_dictionary[i] = []
            # get a list of all the nodes higher up in the dictionary
            nodes_one_level_higher = node_dictionary[i-1]
            for parent in nodes_one_level_higher:
                if parent.action is None:
                    node_dictionary[i].append(parent.left_child)
                    node_dictionary[i].append(parent.right_child)
        
        self.directory = node_dictionary


    def evaluate(self, inputs):
        return self.root_node.evaluate(inputs)
    
# used for mutation if node is currently a terminal leaf
def generate_new_decision_node():
    rng = np.random.default_rng()
    decision_var, range = random.choice(list(input_threshold_ranges.items()))
    threshold = rng.uniform(low=range[0], high=range[1])

    left_child = Node(action=random.choice(action_set))
    right_child = Node(action=random.choice(action_set))

    return Node(decision_var=decision_var, threshold=threshold, left_child=left_child, right_child=right_child)

# a helper to avoid cloning by reference when mating trees
def copy_node(target_node):
    # base case for recursion
    if target_node.action is not None:
        return Node(action=target_node.action)
    else:
        decision_var = target_node.decision_var
        threshold = target_node.threshold
        left_child = copy_node(target_node.left_child)
        right_child = copy_node(target_node.right_child)
        
        return Node(decision_var=decision_var, threshold=threshold,
                    left_child=left_child, right_child=right_child)

def mate_trees(Tree1, Tree2):
    depth_to_search = min(Tree1.depth, Tree2.depth)
    for d in range(0, depth_to_search):
        tree1_candidates = Tree1.directory[d]
        tree2_candidates = Tree2.directory[d]
        for i, Node1 in enumerate(tree1_candidates):
            for j, Node2 in enumerate(tree2_candidates):
                if Node1.decision_var == Node2.decision_var:
                    pass