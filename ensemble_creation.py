from trees import *
import os

# first strategy: measure wastewater every few days
left_grandchild1 = Node(action="set_vax_rate", action_value=0)
right_grandchild1 = Node(action="set_vax_rate", action_value=0.07)

left_child1 = Node(decision_var="current_wes", threshold=0.1,
                    left_child=left_grandchild1, right_child=right_grandchild1)
right_child1 = Node(action="wes_measurement")

root_1 = Node(decision_var="time_since_wes", threshold=0.01, left_child=left_child1, right_child=right_child1)
tree_1 = DecisionTree(root_node=root_1)

# second strategy: measure diagnostics every few days
left_grandchild2 = Node(action="set_vax_rate", action_value=0)
right_grandchild2 = Node(action="set_vax_rate", action_value=0.07)

left_child2 = Node(decision_var="current_diag", threshold=0.1,
                    left_child=left_grandchild2, right_child=right_grandchild2)
right_child2 = Node(action="diagnostic_measurement")

root_2 = Node(decision_var="time_since_diag", threshold=0.01, left_child=left_child2, right_child=right_child2)
tree_2 = DecisionTree(root_node=root_2)

# third strategy: measure diagnostics once in a while
left_grandchild3 = Node(action="set_vax_rate", action_value=0)
right_grandchild3 = Node(action="set_vax_rate", action_value=0.07)

left_child3 = Node(decision_var="current_diag", threshold=0.1,
                    left_child=left_grandchild3, right_child=right_grandchild3)
right_child3 = Node(action="diagnostic_measurement")

root_3 = Node(decision_var="time_since_diag", threshold=0.1, left_child=left_child3, right_child=right_child3)
tree_3 = DecisionTree(root_node=root_3)

# fourth strategy: measure wes every once in a while
left_grandchild4 = Node(action="set_vax_rate", action_value=0)
right_grandchild4 = Node(action="set_vax_rate", action_value=0.07)

left_child4 = Node(decision_var="current_wes", threshold=0.1,
                    left_child=left_grandchild4, right_child=right_grandchild4)
right_child4 = Node(action="wes_measurement")

root_4 = Node(decision_var="time_since_wes", threshold=0.01, left_child=left_child4, right_child=right_child4)
tree_4 = DecisionTree(root_node=root_4)

# put trees in a list for cleaner logic below
ensemble_list = [tree_1, tree_2, tree_3, tree_4]

# save ensemble to a folder to be loaded into a run
cwd = os.getcwd()
subfolder = os.path.join(cwd, "measles_vacc", "custom_starting_ensembles", "first_handcrafted", "round_0")
for i, curr_tree in enumerate(ensemble_list):
    save_to = os.path.join(subfolder, f"tree_{i}.pkl")
    curr_tree.save(save_to=save_to)