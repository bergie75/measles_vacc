# all values controlled by user should come from here
from parameters import exported_parameters
# numerical methods
import numpy as np
import random
import copy
# for saving and loading decision trees
import os
import pickle
# for visualizing trees
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, action=None, action_value = None, in_patch = 0,
                  decision_var=None, threshold=None,
                  left_child=None, right_child=None):
        
        self.action = action
        self.action_value = action_value
        self.in_patch = in_patch
        self.decision_var = decision_var
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.number_calls = 0

    def __repr__(self):
        if self.action is not None:
            if self.action_value is not None:
                return f"{self.action}: {self.action_value} in {self.in_patch}"
            else:
                return self.action
        elif (self.threshold is not None) and (self.decision_var is not None):
            return f"{self.decision_var} in {self.in_patch} <= {self.threshold:0.3f}"
        else:
            return "ERROR: malformed node"

    def terminal_leaf(self):
        return (self.action is not None)
    
    def mutate_patch(self, local_params=exported_parameters):
        num_patches = local_params["num_patches"]
        self.in_patch = np.random.choice(range(0, num_patches))
    
    # defaults to taking a new value between 80% and 120% of old value
    def mutate_threshold(self, local_params=exported_parameters):
        rng = np.random.default_rng()
        max_perc_change = local_params["max_mutate_perc"]
        multiplier = rng.uniform(low=1-max_perc_change, high=1+max_perc_change)
        self.threshold = min(self.threshold*multiplier, 1)
    
    def mutate_decision_var(self, local_params=exported_parameters):
        simulation_outputs = local_params["simulation_outputs"]
        self.decision_var = random.choice(simulation_outputs)
    
    def mutate_action_value(self, local_params=exported_parameters):
        rng = np.random.default_rng()
        max_perc_change = local_params["max_mutate_perc"]
        multiplier = rng.uniform(low=1-max_perc_change, high=1+max_perc_change)
        self.action_value = min(self.action_value*multiplier, 1)
    
    # transforms a node from a leaf to a decision node with two children
    # can be used to grow a tree
    def mutate_into_decision_node(self, local_params=exported_parameters):
        num_patches = local_params["num_patches"]
        simulation_outputs = local_params["simulation_outputs"]
        use_beta_generation = local_params["use_beta_generation"]
        dist_alpha = local_params["dist_alpha"]
        dist_beta = local_params["dist_beta"]
        action_set=local_params["action_set"]
        actions_requiring_values = local_params["actions_requiring_values"]

        rng = np.random.default_rng()
        decision_var = random.choice(simulation_outputs)

        # checks if threshold generation should be skewed
        if use_beta_generation:
            threshold = rng.beta(dist_alpha, dist_beta)
        else:
            threshold = rng.uniform()

        # create children, checking if an action value is needed
        left_child = Node(action=random.choice(action_set), in_patch=np.random.choice(range(0, num_patches)))
        if left_child.action in actions_requiring_values:
            left_child.__setattr__("action_value", rng.uniform())
        
        right_child = Node(action=random.choice(action_set), in_patch=np.random.choice(range(0, num_patches)))
        if right_child.action in actions_requiring_values:
            right_child.__setattr__("action_value", rng.uniform())

        self.action = None
        self.action_value = None
        self.decision_var = decision_var
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

    def mutate_into_leaf(self, local_params=exported_parameters):
        # unpack
        action_set = local_params["action_set"]
        actions_requiring_values = local_params["actions_requiring_values"]

        # random action
        action = random.choice(action_set)

        self.action = action
        if action in actions_requiring_values:
            rng = np.random.default_rng()
            self.action_value = rng.uniform()
        
        self.decision_var = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
    
    def chain_mutation(self, patch_prob, thresh_prob, decision_prob, chop_prob, var_change_prob,
                        action_prob, max_perc_change, depth_remaining):
        rng = np.random.default_rng()
        
        # check for patch change mutation
        if rng.uniform() < patch_prob:
            self.mutate_patch()
        
        # if node is a leaf, potentially turn into a decision node. Checks if depth requirements
        # are violated so that the depth of a tree can be capped. Check leaves first so as not to undo
        # a chop
        if self.action_value is not None and (rng.uniform() < action_prob):
            self.mutate_action_value()
        
        if self.action is not None and (depth_remaining > 0) and (rng.uniform() < decision_prob):
            self.mutate_into_decision_node()
        
        # if node is a decision node, potentially mutate threshold, decision variable, or turn into a leaf
        if self.threshold is not None:
            if rng.uniform() > chop_prob:
                if rng.uniform() < thresh_prob:
                    self.mutate_threshold()
                if rng.uniform() < var_change_prob:
                    self.mutate_decision_var()
            else:
                self.mutate_into_leaf()
          
        # check for children that are not None. If children exist, determine their mutations
        if self.left_child is not None:
            self.left_child.chain_mutation(patch_prob, thresh_prob, decision_prob, chop_prob, var_change_prob,
                        action_prob, max_perc_change, depth_remaining-1)
        if self.right_child is not None:
            self.right_child.chain_mutation(patch_prob, thresh_prob, decision_prob, chop_prob, var_change_prob,
                        action_prob, max_perc_change, depth_remaining-1)
    
    def trim_node(self, local_params=exported_parameters):
        #unpack
        actions_requiring_values = local_params["actions_requiring_values"]

        # checks to see if both actions recommended in node are the same
        # if yes, the check is pointless and the node becomes a leaf
        if self.left_child is not None and self.right_child is not None:
            if (self.left_child.action is not None) and (self.left_child.action == self.right_child.action) and (self.left_child.in_patch == self.right_child.in_patch) and not (self.left_child.action in actions_requiring_values):
                self.action = self.left_child.action
                self.in_patch = self.left_child.in_patch
                self.decision_var = None
                self.threshold = None
                self.left_child = None
                self.right_child = None
        
        # continue down the tree making corrections
        if self.left_child is not None:
            self.left_child.trim_node()
        
        if self.right_child is not None:
            self.right_child.trim_node()
    
    # inputs will be in the form of a dictionary {str(name_decision_var): float(value_of_the_variable)}
    def evaluate(self, inputs, dec_path=""):
        # terminal condition to end the recursion
        if self.terminal_leaf():
            modded_dec_path = dec_path + f"_[{self.action} in {self.in_patch}: {self.action_value}]"
            return self.action, self.action_value, self.in_patch, modded_dec_path
        
        # check threshold to determine the branch to follow, uses dictionary structure of inputs
        if inputs[self.decision_var][self.in_patch] <= self.threshold:
            modded_dec_path = dec_path + f"_[{self.decision_var} in {self.in_patch}<={self.threshold:.2f}]"
            return self.left_child.evaluate(inputs, dec_path=modded_dec_path)
        else:
            modded_dec_path = dec_path + f"_[{self.decision_var} in {self.in_patch}>{self.threshold:.2f}]"
            return self.right_child.evaluate(inputs, dec_path=modded_dec_path)
    
    def reset_calls_recursive(self):
        self.__setattr__("number_calls", 0)
        if self.left_child is not None:
            self.left_child.reset_calls_recursive()
        if self.right_child is not None:
            self.right_child.reset_calls_recursive()
    
    def get_depth(self):
        # base case of recursion
        if self.action is not None:
            return 1
        elif self.decision_var is None:
            return -1
        
        return 1 + max(self.left_child.get_depth(), self.right_child.get_depth())

class DecisionTree:
    def __init__(self, root_node=None, ID_length=128):
        self.root_node = root_node
        # training score stored as a property for quick checks, initialize at maximum cost
        # also holds a standard deviation attribute
        self.training_score = np.inf
        self.score_std = 0
        # ID number used to quickly check if a tree has persisted in training
        self.ID = random.getrandbits(ID_length)
        # calculates the current depth of the tree and initializes
        # a coordinate representation of the tree for later use
        self.update_directory_and_depth()
    
    def evaluate(self, inputs):
        return self.root_node.evaluate(inputs)
    
    def update_directory_and_depth(self):
        new_depth = self.root_node.get_depth()
        node_dictionary = {0: [self.root_node]}
        for i in range(1, new_depth):
            node_dictionary[i] = []
            nodes_one_level_higher = node_dictionary[i-1]
            for parent in nodes_one_level_higher:
                if parent.action is None:
                    node_dictionary[i].append(parent.left_child)
                    node_dictionary[i].append(parent.right_child)
        
        self.directory = node_dictionary
        self.depth = new_depth
    
    def set_by_coordinates(self, depth, index, attribute_name, attribute_value):
        # the part that actually changes the nodes that make up the tree
        self.directory[depth][index].__setattr__(attribute_name, attribute_value)

        # if children are changed, the directory needs to be recalculated. May be more efficient way
        # to do this. Currently reuses code from initialization
        if attribute_name == "left_child" or attribute_name == "right_child":
            self.update_directory_and_depth()
    
    def mutate_tree(self, patch_prob, thresh_prob, decision_prob, chop_prob, var_change_prob,
                        action_prob, max_perc_change, depth_remaining):
        
        self.root_node.chain_mutation(patch_prob, thresh_prob, decision_prob, chop_prob, var_change_prob,
                        action_prob, max_perc_change, depth_remaining)
        self.update_directory_and_depth()
    
    def reset_calls(self):
        self.root_node.reset_calls_recursive()

    def trim_tree(self):
        self.root_node.trim_node()
        self.update_directory_and_depth()
    
    def save(self, save_to):
        with open(save_to, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

# a helper to avoid cloning by reference when mating trees
def copy_node(target_node):
    # base case for recursion
    if target_node.action is not None:
        return Node(action=target_node.action, action_value=target_node.action_value, in_patch=target_node.in_patch)
    else:
        decision_var = target_node.decision_var
        threshold = target_node.threshold
        in_patch = target_node.in_patch
        left_child = copy_node(target_node.left_child)
        right_child = copy_node(target_node.right_child)
        
        return Node(decision_var=decision_var, threshold=threshold, in_patch=in_patch,
                    left_child=left_child, right_child=right_child)

def copy_tree(target_tree):
    return DecisionTree(root_node=copy_node(target_tree.root_node))  # to avoid cloning by reference

# finds possible points where trees can exchange branches
def find_mating_points(Tree1, Tree2):
    min_depth = min(Tree1.depth, Tree2.depth)
    mating_points = []
    
    for d in range(0, min_depth):
        for index1, Node1 in enumerate(Tree1.directory[d]):
            dec_var = Node1.decision_var
            in_patch = Node1.in_patch
            # exclude leaf notes from list of possible joins
            if dec_var is not None:
                mating_points += [[d, index1, index2] for index2, Node2 in enumerate(Tree2.directory[d]) if (Node2.decision_var==dec_var and Node2.in_patch==in_patch)]
    
    return mating_points

# assumes you are handing a valid branch point from both trees parametrized
# by depth and indices, e.g. by calling find_mating_points and selecting
# a random results
def mate_trees_at_coordinates(Tree1, Tree2, depth, index1, index2):
    # trees must be new objects to avoid reference mistakes
    Variant1 = copy_tree(Tree1)
    Variant2 = copy_tree(Tree2)
    Variant3 = copy_tree(Tree1)
    Variant4 = copy_tree(Tree2)

    mean_threshold = (Tree1.directory[depth][index1].threshold + Tree2.directory[depth][index2].threshold)/2

    # the actual nodes used to change the trees can be reused from their parent objects
    # since I haven't been deleting anything yet. This may need to change

    # give variant one the right branch from tree 2, and set the threshold to the mean
    Variant1.set_by_coordinates(depth, index1, "right_child", copy_node(Tree2.directory[depth][index2].right_child))
    Variant1.set_by_coordinates(depth, index1, "threshold", mean_threshold)

    # give variant 2 the left branch of tree one, and set the threshold to the mean
    Variant2.set_by_coordinates(depth, index2, "left_child", copy_node(Tree1.directory[depth][index1].left_child))
    Variant2.set_by_coordinates(depth, index2, "threshold", mean_threshold)

    # give variant 3 the left branch from tree 2, and set the threshold to the mean
    Variant3.set_by_coordinates(depth, index1, "left_child", copy_node(Tree2.directory[depth][index2].left_child))
    Variant1.set_by_coordinates(depth, index1, "threshold", mean_threshold)

    # give variant 4 the right branch of tree one, and set the threshold to the mean
    Variant2.set_by_coordinates(depth, index2, "right_child", copy_node(Tree1.directory[depth][index1].right_child))
    Variant2.set_by_coordinates(depth, index2, "threshold", mean_threshold)

    return Variant1, Variant2, Variant3, Variant4

# uses the preceeding two functions to mate two trees. Note that there will be a bias towards mating
# trees using nodes closer to the leaves, even though that hasn't been explicitly programmed.
# Choosing a mate point uniformly at random causes this, since there will be more by chance lower down
def mate_trees(Tree1, Tree2):
    mating_points = find_mating_points(Tree1, Tree2)
    # there might not be any matching points
    if len(mating_points) > 0:
        random_mate_index = np.random.choice(range(0, len(mating_points)))
        random_mate_point = mating_points[random_mate_index]

        return mate_trees_at_coordinates(Tree1, Tree2, *random_mate_point)
    else:
        return Tree1, Tree2

def grow_random_tree(depth, grow_probability=1, local_params=exported_parameters):
    # unpack
    num_patches = local_params["num_patches"]
    action_set = local_params["action_set"]
    actions_requiring_values = local_params["actions_requiring_values"]

    if depth > 1:
        # DO NOT COMBINE THESE TWO LINES. It doesn't work, I don't know why
        root_node = Node(in_patch=np.random.choice(range(0, num_patches)))
        root_node.mutate_into_decision_node()
    else:
        root_node = Node(action=random.choice(action_set), in_patch=np.random.choice(range(0, num_patches)))
        if root_node.action in actions_requiring_values:
            root_node.__setattr__("action_value", rng.uniform())
    
    dec_tree = DecisionTree(root_node=root_node)
    rng = np.random.default_rng()
    while dec_tree.depth < depth:
        current_leaves = dec_tree.directory[dec_tree.depth-1]
        for leaf in current_leaves:
            if rng.uniform() <= grow_probability:
                leaf.mutate_into_decision_node()
                leaf.__setattr__("in_patch", np.random.choice(range(0, num_patches)))
        # only update directory at the end of the loop to save time
        dec_tree.update_directory_and_depth()
    
    return dec_tree

def load_tree(proj_folder="measles_vacc", sub_folder="", filename="default_tree.pkl"):
    cwd = os.getcwd()
    load_from = os.path.join(cwd, proj_folder, sub_folder, filename)
    with open(load_from, 'rb') as inp:
        loaded_tree = pickle.load(inp)
    
    return loaded_tree

# a function for backwards compatibility, adds an in_patch value of 0
# to each node in a tree for single patch trees trained on old code
def add_patch_to_tree(Tree):
    def node_fix(Node):
        Node.__setattr__("in_patch", 0)
        if Node.left_child is not None:
            node_fix(Node.left_child)
        if Node.right_child is not None:
            node_fix(Node.right_child)
        
        return Node
    
    new_root = node_fix(Tree.root_node)
    return DecisionTree(root_node=new_root)

def visualize_tree(Tree, filename, proj_folder="measles_vacc", sub_folder=""):
    starting_point=8*2**Tree.depth
    blank_line = [" "]*max(128, 4*starting_point)
    
    def recursive_build(Node=Tree.root_node, array_to_build=[copy.deepcopy(blank_line)], 
                        depth_to_mod=0, basepoint=starting_point, child_spacing=starting_point):
        node_info = str(Node)
        length = len(node_info)
        array_to_build[depth_to_mod][basepoint-int(length/2):basepoint+length-int(length/2)] = node_info
        if (Node.left_child is not None or Node.right_child is not None) and len(array_to_build)<=depth_to_mod+1:
            array_to_build.append(copy.deepcopy(blank_line))
        if Node.left_child is not None:
            array_to_build=recursive_build(Node.left_child, array_to_build, depth_to_mod+1, basepoint-int(child_spacing/2**(depth_to_mod+1)))
        if Node.right_child is not None:
            array_to_build=recursive_build(Node.right_child, array_to_build, depth_to_mod+1, basepoint+int(child_spacing/2**(depth_to_mod+1)))
        
        return array_to_build
    
    array_to_build = recursive_build()
    cwd = os.getcwd()
    with open(os.path.join(cwd, proj_folder, sub_folder, filename), "w") as f:
        for i,entry in enumerate(array_to_build):
            f.write(f"{i}:{"".join(entry)}\n")
