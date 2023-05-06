"""
This file contains the code that learns, represents, and abstracts recipes.
"""
import re
import numpy as np
import jax.numpy as jnp
from typing import Iterable
from itertools import combinations, product
from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy

# The total number of features in the envrionment
N_FEATURES = 2

class Statement:

    def __init__(self,
                 input_conditions: Iterable,
                 output_condition: dict
                 ):
        self.input_conditions = {
            "X": input_conditions[0],
            "Y": input_conditions[1]
        }
        self.output_condition = output_condition
        self.cached_values = {}

    def check_inputs_apply(self, x, y):
        input_feats = frozendict({"X": x, "Y": y})

        for input_name, input_condition in self.input_conditions.items():
            feats = input_feats[input_name]
            for feat, val in input_condition.items():
                # If the first half of the implementation isn't satisfied, return true
                if feats[feat] != val:
                    return False

        return True

    def check_output_consistent(self, z):
        # if we make it here, the input conditions have been satisfied
        for feat, val in self.output_condition.items():
            if z[feat] != val:
                return False

        # if we make it here, we meet both the output conditions
        return True

    def get_features_and_values(self):
        # get all the features and their values that are mentioned in this statement
        feature_values = {}
        for condition in list(self.input_conditions.values()) + [self.output_condition]:
            for feature, value in condition.items():
                if feature in feature_values:
                    feature_values[feature].add(value)
                else:
                    feature_values[feature] = set((value,))

        return feature_values

    def get_matrix(self, items):

        matrix = np.zeros((len(items) ** 2, len(items)))
        for i, xy in enumerate(product(items, repeat=2)):
            x, y = xy
            for j, z in enumerate(items):
                matrix[i,j] = int(self(x, y, z))

        return jnp.array(matrix)

    def is_abstract(self):
        """Return whether this rule is an abstract rule"""
        for input_name, conds in self.input_conditions.items():
            if len(conds) < N_FEATURES:
                return True

        if len(self.output_condition) < N_FEATURES:
            return True

        return False


    def __call__(self, x, y, z):

        triple = (x, y, z)
        if triple in self.cached_values:
            return self.cached_values[triple]

        inputs_apply = self.check_inputs_apply(x, y)
        if not inputs_apply:
            ret = True
        else:
            ret = self.check_output_consistent(z)

        self.cached_values[triple] = ret
        return ret

    def __str__(self):

        ret = ""
        input_conditions = []
        for x_feat, x_val in self.input_conditions["X"].items():
            input_conditions.append(f"{x_feat}(X) = {x_val}")
        for y_feat, y_val in self.input_conditions["Y"].items():
            input_conditions.append(f"{y_feat}(Y) = {y_val}")

        ret += " and ".join(input_conditions)

        output_conditions = []
        for z_feat, z_val in self.output_condition.items():
            output_conditions.append(f"{z_feat}(Z) = {z_val}")
        ret += " -> " + " and ".join(output_conditions)

        return ret

    def __hash__(self):
        """Hash a rule using the hash of its string"""
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other.__str__()


def check_input_consistency(rule1, rule2):
    """Check if the input conditions of rule1 include those in rule2"""
    for input_name in ("X", "Y"):
        all_features = set(rule1.input_conditions[input_name].keys()) | set(rule2.input_conditions[input_name].keys())
        for feature in all_features:
            if feature in rule1.input_conditions[input_name] and \
               feature in rule2.input_conditions[input_name] and \
               rule1.input_conditions[input_name][feature] != rule2.input_conditions[input_name][feature]:
                return False

    return True

def check_output_consistency(rule1, rule2):
    """Make sure that the outputs of the two rules don't contradict each other"""
    for feature in set(rule1.output_condition.keys()) | set(rule2.output_condition.keys()):
        # check if both talk about the same feature and necessitate different values
        if feature in rule1.output_condition and \
           feature in rule2.output_condition and \
           rule1.output_condition[feature] != rule2.output_condition[feature]:
            return False
    return True

def contradicts(rule1, rule2):
    """Check if rule1 contradicts rule2"""
    # check if the input conditions of rule1 entail the input conditions of rule2
    if check_input_consistency(rule1, rule2):
        return not check_output_consistency(rule1, rule2)
    else:
        return False

def inputs_generalize(rule1, rule2):
    """Check if the inputs to rule1 are more general than the inpnuts to rule2"""
    for input_name in rule1.input_conditions.keys():
        if input_name not in rule2.input_conditions:
            continue
        for feature, value in rule1.input_conditions[input_name].items():
            if feature not in rule2.input_conditions[input_name] or \
               rule2.input_conditions[input_name][feature] != value:
                return False

    return True


def subsumes(rule1, rule2):
    """Check if rule1 subsumes rule2"""
    if rule1.output_condition == rule2.output_condition:
        return inputs_generalize(rule1, rule2)
    else:
        return False

def check_mutual_consistency(
        statements: Iterable[Statement]
):
    """Check that all statements in a list of statements are compatible with each other"""

    # if both input conditions can be satisfied at the same time, then there must be an item that
    for i, statement1 in enumerate(statements):
        for j, statement2 in enumerate(statements):
            if j == i:
                continue
            if contradicts(statement1, statement2):
                return False

    return True

def get_possible_rules():
    # compute all of the possible conditions

    possible_feature_values = {
        "color": ["green", "red", "blue"],
        "shape": ["triangle", "square", "pentagon"]
    }
    possible_conditions = [{}]
    single_feature_rules = defaultdict(list)
    for feature, values in possible_feature_values.items():
        for value in values:
            single_feature_rules[feature].append({feature: value})
    possible_conditions.extend(single_feature_rules["color"])
    possible_conditions.extend(single_feature_rules["shape"])
    for combo in product(*single_feature_rules.values()):
        possible_conditions.append(combo[0] | combo[1])

    possible_rules = []
    for rule_conditions in product(possible_conditions, repeat=3):
        x_cond, y_cond, z_cond = rule_conditions
        if (len(x_cond) == 2 and len(y_cond) == 2) or len(z_cond) == 0:
            continue
        possible_rules.append(Statement([x_cond, y_cond], z_cond))

    return possible_rules

def abstract(
        knowledge_base: Iterable[Statement],
        known_items: Iterable[dict]
):
    """
    Infer generalizations about recipes. Returns a list of warranted abstractions.
    """

    possible_rules = get_possible_rules()

    # iterate over possible rules, keeping only those that don't contradict any known rules
    valid_abstractions = set()
    for proposed_rule in possible_rules:
        contradicts_known_rules = False
        pos_examples = 0
        for known_rule in knowledge_base:
            is_pos_example = False
            if check_input_consistency(proposed_rule, known_rule):
                if check_output_consistency(proposed_rule, known_rule):
                    pos_examples += 1
                else:
                    contradicts_known_rules = True
                    break

            if is_pos_example:
                pos_examples += 1
            if contradicts_known_rules:
                break

        if pos_examples >= 2 and not contradicts_known_rules:
            valid_abstractions.add(proposed_rule)

    return valid_abstractions

def anti_unify(statements):
    """
    Compute the least-general generalization of a set of statements
    """
    # aggregate all the output conditions
    total_output_condition = {}
    for statement in statements:
        for feature, value in statement.output_condition.items():
            if feature in total_output_condition:
                if total_output_condition[feature] != value:
                    return None
            else:
                total_output_condition[feature] = value

    general_input_conditions = dict([(cond_name, dict(cond)) for cond_name, cond in statements[0].input_conditions.items()])
    to_remove = []
    for statement in statements:
        for input_name in general_input_conditions.keys():
            for feature, value in general_input_conditions[input_name].items():
                if feature not in statement.input_conditions[input_name] or \
                   statement.input_conditions[input_name][feature] != value:
                    to_remove.append((input_name, feature))

    for input_name, feature in to_remove:
        del general_input_conditions[input_name][feature]

    general_statement = Statement([general_input_conditions["X"], general_input_conditions["Y"]], total_output_condition)

    return general_statement

if __name__ == "__main__":

    known_items = [
        {"color": "green", "shape": "triangle"},
        {"color": "green", "shape": "square"},
        {"color": "green", "shape": "pentagon"},
        {"color": "red", "shape": "triangle"},
        {"color": "red", "shape": "square"},
        {"color": "red", "shape": "pentagon"},
        {"color": "blue", "shape": "triangle"},
        {"color": "blue", "shape": "square"},
        {"color": "blue", "shape": "pentagon"}
    ]

    # learn some recipes
    knowledge = []
    knowledge.append(Statement(
        [
            {"color": "green", "shape": "triangle"},
            {"color": "red", "shape": "square"}
        ],
        {"color": "blue", "shape": "pentagon"}
    ))
    knowledge.append(Statement(
        [
            {"color": "green", "shape": "square"},
            {"color": "red", "shape": "square"}
        ],
        {"color": "blue", "shape": "triangle"}
    ))

    # print the knowledge
    print("\nbackground knowledge:")
    print(knowledge[0])
    print(knowledge[1])

    print("\n")

    # generate some abstractions
    abstractions = abstract(knowledge, known_items)

    print(f"{len(abstractions)} consistent abstractions:")
    for abstr in abstractions:
        print(abstr)

    print("\n")

    # test contradiction
    rule1 = Statement(
        input_conditions=[{"color": "red"}, {"shape": "square"}],
        output_condition={"color": "blue"}
    )

    rule2 = Statement(
        input_conditions=[{"color": "red"}, {"color": "green"}],
        output_condition={"color": "red"}
    )

    breaking_input = {'color': 'red', 'shape': 'triangle'}, {'color': 'green', 'shape': 'square'}

    print(contradicts(rule1, rule2))

    # test anti-unification

