"""
This file contains the base code for CultuRL games
"""
from copy import deepcopy
from typing import Callable, Optional, Union
from frozendict import frozendict
from pyprojroot import here
import numpy as np


class State:

    def __init__(self,
                 inventory: list,
                 ):
        self.inventory = inventory.copy()

    def __str__(self):
        response = "inventory: "
        response += str(self.inventory)
        return response

    def __bool__(self):
        return self.is_end()

    def _key(self):
        return frozenset(self.inventory)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, State):
            return self._key() == other._key()
        else:
            raise TypeError("Cannot compare State to {}".format(type(other)))

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    def is_end(self):
        return len(self.inventory) == 0


class NoSellState(State):

    def is_end(self):
        return len(self.inventory) == 1 and self.inventory[0]["n"] == 1

class MinCraftState(State):

    def __init__(self,
                 inventory: list,
                 n_crafts: int
                 ):
        super().__init__(inventory)
        self.n_crafts = n_crafts

class Game:

    def __init__(self,
                 starting_inventory: list,
                 recipe_fn: Callable,
                 value_fn: Callable,
                 ):

        self.starting_inventory = starting_inventory
        self.value_fn = value_fn
        self.recipe_fn = recipe_fn

    def get_start_state(self):

        return State(
            inventory=self.starting_inventory,
        )

    def update_inventory(
            self,
            curr_inventory,
            items: list,
            values: list
    ):
        # check for duplicate items:
        item_updates = {}
        for i, v in zip(items, values):
            if i not in item_updates:
                item_updates[i] = 0
            item_updates[i] += v

        new_inventory = []
        curr_features = [x["features"] for x in curr_inventory]
        for item, value in item_updates.items():
            if item not in curr_features:
                new_inventory.append(frozendict({"features": item, "n": value}))
        for curr_item in curr_inventory:
            update_this_one = False
            for item, value in item_updates.items():
                # update the item if it exists in the inventory
                if curr_item["features"] == item:
                    update_this_one = True
                    break
            if update_this_one:
                if curr_item["n"] + value > 0:
                    new_inv_item = frozendict({
                        "n": curr_item["n"] + value,
                        "features": curr_item["features"]
                    })
                    new_inventory.append(new_inv_item)
                elif curr_item["n"] + value < 0:
                    raise ValueError("tried to remove more of an item than exists!")
            else:
                new_inventory.append(curr_item)

        return new_inventory

    def process_craft(self, state, first_idx, second_idx):

        # get the items
        item1, item2 = state.inventory[first_idx], state.inventory[second_idx]
        # compute the output
        output = self.recipe_fn(item1, item2)
        # remove the two current items from the inventory
        new_inventory = self.update_inventory(state.inventory, [item1["features"], item2["features"], output["features"]], [-1, -1, 1])
        # create the next state
        new_state = State(new_inventory)

        return new_state, 0

    def get_sell_value(self, item):
        """get the sell value for a particular item"""
        return item["n"] * self.value_fn(item)

    def get_total_sell_value(self, state):
        """get the sum of the sell values of everything"""
        return sum([self.get_sell_value(item) for item in state.inventory])

    def process_sell(self, state):
        """
        Process the decision to sell all items
        """

        # compute the reward
        sum_value = sum([self.get_sell_value(x) for x in state.inventory])

        # the next state has an empty inventory
        new_state = State(inventory = [])

        return new_state, sum_value

    def get_actions(self, state):
        """
        Get a list of legal actions, where:
        0:n-1 - add the given item to the crafting bench
        n: - sell all items in the inventory
        :param state:
        :return:
        """
        # we can always sell our current inventory (action 0)
        legal_actions = [0]

        # alternatively, we can craft any combination of two resources on the crafting bench.
        n_unique_items = len(state.inventory)
        for i in range(n_unique_items):
            for j in range(n_unique_items):
                if j > i or (j == i and state.inventory[i]["n"] == 1):
                    break
                # add this pair to the set of possible actions
                legal_actions.append(i * n_unique_items + j + 1)


        return legal_actions

    def take_action(self, state, action):

        if isinstance(action, np.int64):
            action = int(action)
        # parse the action: get the type and relevant item
        if not isinstance(action, int):
            raise RuntimeError(f"Only integer actions are allowed. Got type: {type(action)}")

        if action == 0:
            return self.process_sell(state)
        else:
            n_unique_items = len(state.inventory)
            # unpack the items to be crafted
            first_idx = (action - 1) // n_unique_items
            second_idx = (action - 1) % n_unique_items
            return self.process_craft(state, first_idx, second_idx)


class NoSellGame(Game):
    """
    A variant of the crafting game where you can't sell anything.
    You just have to keep crafting things together until you're left with one item.
    """

    def get_start_state(self):

        return NoSellState(
            inventory=self.starting_inventory,
        )

    def get_actions(self, state):
        """
        Get a list of legal actions, where:
        0:n-1 - add the given item to the crafting bench
        n: - sell all items in the inventory
        :param state:
        :return:
        """
        legal_actions = []
        # alternatively, we can craft any combination of two resources on the crafting bench.
        n_unique_items = len(state.inventory)
        for i in range(n_unique_items):
            for j in range(n_unique_items):
                if j > i or (j == i and state.inventory[i]["n"] == 1):
                    break
                # add this pair to the set of possible actions
                legal_actions.append(i * n_unique_items + j)

        return legal_actions

    def take_action(self, state, action):

        if isinstance(action, np.int64):
            action = int(action)
        # parse the action: get the type and relevant item
        if not isinstance(action, int):
            raise RuntimeError(f"Only integer actions are allowed. Got type: {type(action)}")

        n_unique_items = len(state.inventory)
        # unpack the items to be crafted
        first_idx = action // n_unique_items
        second_idx = action % n_unique_items
        new_state, reward = self.process_craft(state, first_idx, second_idx)
        new_state = NoSellState(new_state.inventory)
        if new_state.is_end():
            reward = self.get_reward(new_state)
        return new_state, reward

    def get_reward(self, state):
        """
        compute the reward of a terminal state
        """
        remaining_item = state.inventory[0]
        return self.value_fn(remaining_item)

class MinCraftGame(Game):

    def __init__(self,
                 starting_inventory: list,
                 min_crafts: int,
                 recipe_fn: Callable,
                 value_fn: Callable,
                 ):
        super().__init__(starting_inventory,
                         recipe_fn,
                         value_fn)
        self.min_crafts = min_crafts

    def get_start_state(self):

        return MinCraftState(
            inventory=self.starting_inventory,
            n_crafts=0
        )

    def get_actions(self, state: MinCraftState):
        """
        Get a list of legal actions, where:
        0:n-1 - add the given item to the crafting bench
        n: - sell all items in the inventory
        :param state:
        :return:
        """
        # we can always sell our current inventory (action 0)
        legal_actions = []
        if state.n_crafts >= self.min_crafts:
            legal_actions.append(0)

        # alternatively, we can craft any combination of two resources on the crafting bench.
        n_unique_items = len(state.inventory)
        for i in range(n_unique_items):
            for j in range(n_unique_items):
                if j > i or (j == i and state.inventory[i]["n"] == 1):
                    break
                # add this pair to the set of possible actions
                legal_actions.append(i * n_unique_items + j + 1)


        return legal_actions

    def take_action(self, state: MinCraftState, action):

        if isinstance(action, np.int64):
            action = int(action)
        # parse the action: get the type and relevant item
        if not isinstance(action, int):
            raise RuntimeError(f"Only integer actions are allowed. Got type: {type(action)}")

        if action == 0:
            new_state, reward =  self.process_sell(state)
            min_craft_state = MinCraftState(new_state.inventory, state.n_crafts)
            return min_craft_state, reward
        else:
            n_unique_items = len(state.inventory)
            # unpack the items to be crafted
            first_idx = (action - 1) // n_unique_items
            second_idx = (action - 1) % n_unique_items
            new_state, reward = self.process_craft(state, first_idx, second_idx)
            min_craft_state = MinCraftState(new_state.inventory, state.n_crafts + 1)
            return min_craft_state, reward

class EpisodicState(State):

    def __init__(self,
                 inventory: list,
                 goal: frozendict
                 ):
        super().__init__(inventory)
        self.goal = goal

    def __str__(self):
        response = "inventory: "
        response += str(self.inventory)
        response += f"\ngoal: {self.goal}"
        return response

    def _key(self):
        return frozenset(self.inventory + [str(self.goal)])

    def is_goal_state(self):
        for item in self.inventory:
            if item["features"] == self.goal:
                return True
        return False

    def is_end(self):
        if len(self.inventory) == 1 and self.inventory[0]["n"] == 1:
            return True
        else:
            return self.is_goal_state()

class EpisodicGame(Game):

    def __init__(self,
                 starting_inventory: list,
                 possible_items: list,
                 recipe_fn: Callable
                 ):
        super().__init__(starting_inventory,
                         recipe_fn,
                         lambda x: 0)
        starting_features = [x["features"] for x in starting_inventory]
        self.possible_goals = [i for i in possible_items if i not in starting_features]

    def get_start_state(self):

        goal = np.random.choice(self.possible_goals)

        return EpisodicState(
            inventory=self.starting_inventory,
            goal=goal
        )

    def get_actions(self, state):
        """
        Get a list of legal actions, where:
        0:n-1 - add the given item to the crafting bench
        n: - sell all items in the inventory
        :param state:
        :return:
        """
        # we can never sell anything, the only action is to craft
        legal_actions = []

        # alternatively, we can craft any combination of two resources on the crafting bench.
        n_unique_items = len(state.inventory)
        for i in range(n_unique_items):
            for j in range(n_unique_items):
                if j > i or (j == i and state.inventory[i]["n"] == 1):
                    break
                # add this pair to the set of possible actions
                legal_actions.append(i * n_unique_items + j + 1)

        return legal_actions

    def take_action(self, state, action):

        if isinstance(action, np.int64):
            action = int(action)
        # parse the action: get the type and relevant item
        if not isinstance(action, int):
            raise RuntimeError(f"Only integer actions are allowed. Got type: {type(action)}")

        n_unique_items = len(state.inventory)
        # unpack the items to be crafted
        first_idx = (action - 1) // n_unique_items
        second_idx = (action - 1) % n_unique_items
        new_state, reward = self.process_craft(state, first_idx, second_idx)
        # turn the output into an episodic state with the same goal
        new_episodic_state = EpisodicState(inventory=new_state.inventory, goal=state.goal)
        return new_episodic_state, reward

class RandomEpisodicGame(EpisodicGame):

    def __init__(self,
                 possible_items: list,
                 recipe_fn: Callable
                 ):
        super().__init__([], possible_items, recipe_fn)
        self.possible_items = possible_items

    def get_start_state(self):

        # initialize with one of each item
        starting_items = np.random.choice(self.possible_items, size=3, replace=False)
        starting_inventory = [{"n": 1, "features": feats} for feats in starting_items]
        for i in range(3):
            inv_item = np.random.choice(starting_inventory)
            inv_item["n"] += 1
        starting_inventory = [frozendict({"n": x["n"], "features": frozendict(x["features"])}) for x in starting_inventory]

        possible_goals = [i for i in self.possible_items if i not in starting_items]
        goal = np.random.choice(possible_goals)

        return EpisodicState(
            inventory=starting_inventory,
            goal=goal
        )
