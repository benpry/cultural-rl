"""
A model of task solving and cultural transmission
"""
import re
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from game import Game, State, NoSellGame
from random import random, sample
from typing import Callable, Optional, Iterable
from itertools import product, combinations
from frozendict import frozendict
from pyprojroot import here
from tqdm import tqdm
from recipe_learning import Statement, abstract, check_mutual_consistency, contradicts, subsumes, anti_unify
from psrl import get_psrl_actions, DSLSampler, MemoizedWorldModel, TheoryBasedSampler

N_FEATURES = 2

def stochastic_argmax(lst):
    """
    Return the argmax if a unique one exists. If not, choose randomly among the values that are tied for best.
    """
    am = np.argmax(lst)
    if isinstance(am, np.int64):
        return am
    else:
        return sample(am, 1)[0]

def message_world_model(x, y, z, message):
    """
    A world model that checks if x and y can craft together to make z.
    """
    # check all the learned constraints
    for statement in message:
        if not statement(x, y, z):
            return False

    # return true by default
    return True

# @jax.jit
def compute_divergences(
        candidate_message_matrices: jnp.array,
        world_model_matrix: jnp.array
        ):
    """
    compute the divergences between the world model and candidate message vectors 
    """
    world_model_dist = world_model_matrix / jnp.expand_dims(jnp.sum(world_model_matrix, axis=-1), -1)
    candidate_message_dists = candidate_message_matrices / jnp.expand_dims(jnp.sum(candidate_message_matrices, axis=-1), -1)
    message_divergences = jnp.abs(candidate_message_dists - world_model_dist).sum(axis=-1).mean(axis=-1)

    return message_divergences

# @jax.jit
def find_best_statement_combo(world_model_vector: jnp.array,
                              all_statements_matrix: jnp.array,
                              combos_to_try: jnp.array):

    message_vector_combos = all_statements_matrix[combos_to_try, :]
    message_vector_combos = message_vector_combos.min(axis=1)
    message_divergences = compute_divergences(message_vector_combos, world_model_vector)

    best_combo_idx = jnp.argmin(message_divergences)
    best_divergence = jnp.min(message_divergences)
    return best_combo_idx, best_divergence


class CulturalTransmissionAgent:
    """
    A model of one individual in a transmission chain setup
    """

    def __init__(
            self,
            channel_capacity: Optional[int] = None,
    ):
        self.channel_capacity = channel_capacity
        self.memoized_world_model = None
        self.psrl_actions = None

    def compute_policy(self, game, state):
        """Do PSRL to get a policy"""
        world_model = self.get_memoized_world_model()
        _, actions, _ = get_psrl_actions(
            game,
            start_state=state,
            world_model=world_model,
        )
        self.psrl_actions = actions

    def solution_policy(self, game, state):
        """Return the action to take in a particular state"""
        if self.psrl_actions is None:
            self.compute_policy(game, state)
        return self.psrl_actions.pop(0)

    def get_world_model_matrix(self):
        """
        Turn the world model into a vector
        """
        all_matrices = []
        for statement in self.knowledge:
            all_matrices.append(statement.get_matrix(self.possible_items))
        world_model_matrix = jnp.stack(all_matrices).astype(jnp.int8)

        return jnp.min(world_model_matrix, axis=0)

    def solve_task(self, game) -> int:
        """
        Solve the task given the current knowledge and solution policy
        """
        #FIXME: at the moment can't run this in isolation because reset 
        # is needed to reset the memoized world model. called in full_loop.
        state = game.get_start_state()
        self.start_episode(game, state)
        total_score = 0
        while not state.is_end():
            action = self.solution_policy(game, state)
            # print(f"*** Taking action {action} ***")
            new_state, reward = game.take_action(state, action)
            self.update_knowledge(game, state, action, new_state)
            total_score += reward
            state = new_state

        return total_score

    def demonstrate(self, game, n=10):
        """
        Complete n episodes of the task without learning anything and return
        the mean score
        """
        scores = []
        for i in range(n):
            state = game.get_start_state()
            total_score = 0
            while not state.is_end():
                self.compute_policy(game, state)
                action = self.solution_policy(game, state)
                new_state, reward = game.take_action(state, action)
                total_score += reward
                state = new_state
            scores.append(total_score)

        return np.mean(scores)


    def full_loop(self, task, message = (), transmit: bool = True):
        """
        Receive a message (if one is specified), then work on the task, then transmit a message for the next generation
        """
        self.reset() #reset first, so memoized world model is reset
        self.receive_message(message)
        score = self.solve_task(task)
        if transmit:
            new_message = self.transmit_message(
                task,
                channel_capacity=self.channel_capacity
            )
        else:
            new_message = None
        
        return score, new_message

    def get_memoized_world_model(self) -> MemoizedWorldModel:
        """Returns the current memoized world model"""
        return self.memoized_world_model

    def get_sampler(self) -> TheoryBasedSampler:
        """Return the sampler"""
        raise NotImplementedError()

    def reset(self):
        """Reset the internal state, to be done after running a loop"""
        self.reset_knowledge()
        self.sampler = self.get_sampler()
        self.memoized_world_model = MemoizedWorldModel(self.sampler)

    def start_episode(self, game, start_state):
        """Do anything necessary when starting an episode"""
        pass
    
    def reset_knowledge(self):
        """Reset the knowledge"""
        raise NotImplementedError()

    def receive_message(self, message):
        """For now, we'll assume the message is a list of DSL statements that we copy into our knowledge"""
        raise NotImplementedError()

    def update_knowledge(self, game, old_state, action, new_state):
        """
        Update the current theory based on the fact that we transitioned from
        old_state to new_state when we took the given action
        """
        raise NotImplementedError()

    def transmit_message(
            self,
            game: Game,
            channel_capacity: Optional[int] = None
    ):
        raise NotImplementedError()

class DSLLogicalAgent(CulturalTransmissionAgent):
    """
    An agent that solves the task using a world model based on DSL statements
    """

    def __init__(
            self,
            channel_capacity: Optional[int] = None,
            can_abstract: bool = True,
            message_objective: str = "divergence",
            message_selection_method: str = "decremental",
            init_possible_items: Iterable[dict] = (),
            use_cuda: bool = True
    ):
        super().__init__(channel_capacity=channel_capacity)

        self.can_abstract = can_abstract
        self.message_objective = message_objective
        self.message_selection_method = message_selection_method
        self.use_cuda = use_cuda

        self.init_possible_items = set([frozendict(item) for item in init_possible_items])
        self.possible_items = self.init_possible_items

        # set of DSL statements
        self.knowledge = set()

    def reset_knowledge(self):
        self.knowledge = set()
        self.possible_items = self.init_possible_items
        # self.sampler = DSLSampler(self.knowledge, self.possible_items)
        # self.memoized_world_model = MemoizedWorldModel(self.sampler)

    def update_possible_items(self, state=None):
        """go through the knowledge and update the items we can infer exist from them"""
        features_with_values = {}

        # we know the features mentioned in
        for statement in self.knowledge:
            feature_vals = statement.get_features_and_values()
            for feature, values in feature_vals.items():
                if feature in features_with_values:
                    features_with_values[feature] |= values
                else:
                    features_with_values[feature] = values

        if state is not None:
            for item in state.inventory:
                for feature in item["features"]:
                    if feature in features_with_values:
                        features_with_values[feature].add(item["features"][feature])
                    else:
                        features_with_values[feature] = set((item["features"][feature],))

        self.features_with_values = features_with_values
        for feats in product(*[list(features_with_values[x]) for x in features_with_values]):
            item = {}
            for i, feat_name in enumerate(features_with_values):
                item[feat_name] = feats[i]
            if len(item) == N_FEATURES:
                self.possible_items.add(frozendict(item))

    def start_episode(self, game, start_state):
        """Update the possible items based on the start state"""
        self.update_possible_items(start_state)

    def receive_message(self, message):
        """Update the agent's knowledge given the evidence in a message"""
        for statement in message:
            self.add_to_knowledge(statement)
        self.update_possible_items()

    def world_model(self, x, y, z):
        """
        A world model that checks if x and y can craft together to make z.
        """
        # check all the learned constraints
        for statement in self.knowledge:
            if not statement(x, y, z):
                return False

        # return true by default
        return True

    def get_sampler(self):
        """Returns a DSL-based sampler"""
        return DSLSampler(self.knowledge, self.possible_items)

    def add_to_knowledge(self, statement):
        """Add the given statement to the knowledge, removing any statements it contradicts"""
        # remove statements that the new statement contradicts
        contradicted_statements = set()
        for existing_statement in self.knowledge:
            if contradicts(statement, existing_statement):
                contradicted_statements.add(existing_statement)

        for statement in contradicted_statements:
            self.knowledge.remove(statement)

        # add the new statement
        self.knowledge.add(statement)

    def update_knowledge(self, game, old_state, action, new_state):
        """Given that we took (action) and transitioned from old_state to new_state, update our knowledge base"""
        if len(new_state.inventory) == 0:
            # if we sold everything, we don't update the knowledge
            return

        n_unique_items = len(old_state.inventory)
        # unpack the items to be crafted
        if isinstance(game, NoSellGame):
            first_idx = action // n_unique_items
            second_idx = action % n_unique_items
        else:
            first_idx = (action - 1) // n_unique_items
            second_idx = (action - 1) % n_unique_items
        x = frozendict(old_state.inventory[first_idx]["features"])
        y = frozendict(old_state.inventory[second_idx]["features"])
        pre_craft_inventory = old_state.inventory

        for craft_item in (x, y):
            for inv_item in pre_craft_inventory:
                if inv_item["features"] == craft_item:
                    break
            # found the matching item, now decrement its count
            pre_craft_inventory.remove(inv_item)
            new_inv_item = frozendict({"n": inv_item["n"] - 1, "features": inv_item["features"]})
            if new_inv_item["n"] > 0:
                pre_craft_inventory.append(new_inv_item)

        z = None
        for new_item in new_state.inventory:
            found_z = False
            found_match = False
            for old_item in pre_craft_inventory:
                if new_item["features"] == old_item["features"]:
                    found_match = True
                    if new_item["n"] > old_item["n"]:
                        z = frozendict(new_item["features"])
                        found_z = True
                        break
            if not found_match:
                z = frozendict(new_item["features"])
                found_z = True
            if found_z:
                break

        knowledge_item = Statement([x, y], z)
        if knowledge_item not in self.knowledge:
            self.add_to_knowledge(knowledge_item)
            self.update_possible_items()
            # recompute the policy if the world model changes
            if self.memoized_world_model.patch(x, y, z):
                self.psrl_actions = None  # psrl actions are no longer valid

    def get_abstractions(self):
        """Given all the knowledge, compute the best abstractions"""
        all_abstractions = abstract(self.knowledge, self.possible_items)

        abstractions = all_abstractions
        # abstractions = set([a for a in all_abstractions if len(a.input_conditions["X"]) > 0 and len(a.input_conditions["Y"]) > 0])

        return abstractions

    def get_lowest_divergence_message(
            self,
            knowledge_base,
            channel_capacity
    ):
        knowledge_base = list(knowledge_base)
        world_model_matrix = self.get_world_model_matrix()
        all_statements_matrix = jnp.stack([statement.get_matrix(self.possible_items) for statement in knowledge_base],
                                          dtype=jnp.int8)
        combos_to_try = jnp.array(np.array(list(combinations(range(len(knowledge_base)), channel_capacity))))

        best_combo_idx, _ = find_best_statement_combo(
            world_model_matrix,
            all_statements_matrix,
            combos_to_try
        )

        best_combo = combos_to_try[best_combo_idx]
        best_message = set([knowledge_base[i] for i in best_combo])

        return best_message

    def get_lowest_divergence_message_decremental(
            self,
            knowledge_base,
            channel_capacity
    ):
        knowledge_base = list(knowledge_base)
        world_model_matrix = self.get_world_model_matrix()
        all_statements_matrix = jnp.stack([statement.get_matrix(self.possible_items) for statement in knowledge_base],
                                          dtype=jnp.int8)

        best_message_by_length = {}
        message_scores_by_length = {}
        for length in reversed(range(1, len(knowledge_base) + 1)):
            if length == len(knowledge_base):
                best_message_by_length[length] = set(range(len(knowledge_base)))
                message_scores_by_length[length] = 0
                continue

            best_larger_message = best_message_by_length[length+1]
            candidate_combos = jnp.array([list(best_larger_message - {i}) for i in best_larger_message])

            best_combo_idx, best_divergence = find_best_statement_combo(
                world_model_matrix,
                all_statements_matrix,
                candidate_combos
            )

            best_combo = candidate_combos[best_combo_idx]
            best_message_by_length[length] = set(best_combo.tolist())
            message_scores_by_length[length] = best_divergence

        best_combo_overall = best_message_by_length[1]
        best_score = 999999
        for i in range(1, channel_capacity + 1):
            if message_scores_by_length[i] < best_score:
                best_combo_overall = best_message_by_length[i]

        best_message = set([knowledge_base[i] for i in best_combo_overall])

        return best_message

    def anti_unify_statement_pairs(self, statements):

        # anti-unify each pair of statements
        anti_unified_statements = []
        for s1, s2 in combinations(statements, 2):
            if s1 == s2:
                continue
            su = anti_unify([s1, s2])
            if su is not None:
                anti_unified_statements.append((s1, s2, su))

        # remove generalizations that contradict existing knowledge
        to_remove = set()
        for s1, s2, su in anti_unified_statements:
            for s_curr in statements:
                if contradicts(su, s_curr):
                    to_remove.add(su)
                    break
        anti_unified_statements = [s for s in anti_unified_statements if s[2] not in to_remove]

        return anti_unified_statements

    def get_most_conservative_generalization(self, generalizations):

        scores = []
        for generalization in generalizations:
            s1, s2, su = generalization
            score = len(su.input_conditions["X"]) + len(su.input_conditions["Y"]) - \
                    max(len(s1.input_conditions["X"]), len(s2.input_conditions["X"])) - \
                    max(len(s1.input_conditions["Y"]), len(s2.input_conditions["Y"]))
            scores.append(score)

        best_gen_idx = stochastic_argmax(scores)
        return generalizations[best_gen_idx]

    def choose_statement_to_remove(self, statements):
        # we will remove the most specific statement
        scores = []
        for statement in statements:
            specificity = len(statement.input_conditions["X"]) + len(statement.input_conditions["Y"]) - len(statement.output_condition)
            redundancy = len([s for s in statements if s.output_condition == statement.output_condition])
            scores.append(100 * specificity + redundancy)

        best_statement_idx = stochastic_argmax(scores)
        return statements[best_statement_idx]

    def get_message_via_anti_unification(self,
                                         statements,
                                         channel_capacity,
                                         ):
        message = list(statements)
        while len(message) > channel_capacity:
            # first, try to anti-unify statements
            if self.can_abstract:
                anti_unified_statements = self.anti_unify_statement_pairs(message)
            else:
                anti_unified_statements = []

            if len(anti_unified_statements) > 0:
                # get the most conservative generalization
                s1, s2, su = self.get_most_conservative_generalization(anti_unified_statements)
                # remove the specific statements and add the unified statement
                message.remove(s1)
                message.remove(s2)
                message.append(su)
            else:
                # if we can't anti-unify, remove something from the knowledge base
                s_remove = self.choose_statement_to_remove(message)
                message.remove(s_remove)

        return message

    def transmit_message(
            self,
            game: Optional[Game] = None,
            channel_capacity: Optional[int] = None,
    ):
        """For now, just return the knowledge exactly (belief copying)"""
        if channel_capacity is None or channel_capacity >= len(self.knowledge):  # None means unlimited channel
            return self.knowledge

        # compute abstractions and merge it into a knowledge base
        if self.can_abstract and self.message_selection_method != "anti_unification":
            abstractions = self.get_abstractions()
        else:
            abstractions = set()
        knowledge_base = self.knowledge | abstractions

        if self.message_selection_method == "decremental":
            message = self.get_lowest_divergence_message_decremental(
                knowledge_base,
                channel_capacity
            )
        elif self.message_selection_method == "anti_unification":
            message = self.get_message_via_anti_unification(
                self.knowledge,
                channel_capacity
            )
        elif self.message_selection_method == "brute_force":
            # run rollouts with each possible subset of the knowledge
            message = self.get_lowest_divergence_message(
                knowledge_base,
                channel_capacity
            )
        else:
            raise ValueError(f"Unrecognized selection method: {self.message_selection_method}")

        return message
