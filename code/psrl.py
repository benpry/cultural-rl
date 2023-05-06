"""
Implementation of Posterior Sampling Reinforcement Learning for planning in a sample from a world model
"""
from frozendict import frozendict
from copy import deepcopy
import numpy as np
from queue import Queue
from game import State, NoSellGame, NoSellState, MinCraftGame, MinCraftState, EpisodicGame, EpisodicState
from recipe_learning import Statement


class TheoryBasedSampler:
    """
    Samples from p(z|x, y, T)
    """
    def __call__(self, x, y):
        """Sample a z (output) given an x and y (inputs)"""
        raise NotImplementedError()

class DSLSampler(TheoryBasedSampler):

    def __init__(
            self,
            knowledge: set,
            known_items: set
    ):
        self.knowledge = knowledge
        self.known_items = known_items

    def __call__(self, x, y):
        """compute p(z|x,y,T) (where T is a theory represented by )"""
        possible_zs = []
        for z in self.known_items:
            violates_knowledge = False
            for statement in self.knowledge:
                if not statement(x, y, z):
                    violates_knowledge = True
                    break
            if not violates_knowledge:
                possible_zs.append(z)

        # sample and memoize a z-value
        z = np.random.choice(possible_zs)
        
        if len(z) == 1:
            print("something weird is going on")

        return z

class MemoizedWorldModel:
    """
    A world model where the z values are memoized.
    This is effectively a sample from the space of world models consistent with a set of knowledge.
    """

    def __init__(self, sampler: TheoryBasedSampler):
        self.sampler = sampler
        self.memory = {}

    def reset(self):
        """Reset the memory"""
        self.memory = {}

    def __call__(self, x, y):
        """
        get a z sample from the memoized model
        """
        sample_dict = frozendict({"x": x, "y": y})
        if sample_dict in self.memory:
            return self.memory[sample_dict]
        z = self.sampler(x, y)
        self.memory[sample_dict] = z

        return z

    def patch(self, x, y, z):
        """
        Update the memory with an observed x,y,z triple.
        Returns True if the patch either adds or changes something in memory
        """
        sample_dict = frozendict({"x": x, "y": y})
        if sample_dict in self.memory and self.memory[sample_dict] == z:
            return False

        self.memory[sample_dict] = z
        return True

def get_next_state_from_world_model(state, action, game, world_model):
    """
    Use our world model to get a next state and a reward
    """
    assert world_model

    if action == 0 and not isinstance(game, NoSellGame):
        sell = game.process_sell(state)
        # print(f"Action 0, sell. value {sell}")
        return sell

    n_unique_items = len(state.inventory)
    # unpack the items to be crafted
    if isinstance(game, NoSellGame):
        first_idx = action // n_unique_items
        second_idx = action % n_unique_items
    else:
        first_idx = (action - 1) // n_unique_items
        second_idx = (action - 1) % n_unique_items
    # get the items
    item1, item2 = state.inventory[first_idx], state.inventory[second_idx]
    output = world_model(item1["features"], item2["features"])
    new_inventory = game.update_inventory(state.inventory, [item1["features"], item2["features"], output], [-1, -1, 1])
    if isinstance(game, NoSellGame):
        next_state = NoSellState(new_inventory)
    elif isinstance(game, MinCraftGame):
        next_state = MinCraftState(new_inventory, n_crafts=state.n_crafts + 1)
    elif isinstance(game, EpisodicGame):
        next_state = EpisodicState(new_inventory, goal=state.goal)
    else:
        next_state = State(new_inventory)
    reward = 0

    if isinstance(game, NoSellGame) and next_state.is_end():
        reward = game.get_reward(next_state)

    # print(f"Action {action}. Crafting {item1['features']['color']} {item1['features']['shape']} and {item2['features']['color']} {item2['features']['shape']} to get {output['color']} {output['shape']}")
    # print(f"Start state: {state}")
    # print(f"End state: {next_state}\n")

    return next_state, reward


def get_optimal_action_sequence(game, start_state, world_model):
    """
    Get a list of all state-action pairs
    """
    assert world_model
    open_q = Queue()
    open_q.put([[start_state], []])
    seen = set((start_state,))

    best_path, best_action_sequence, best_reward = [], [], -999999
    while not open_q.empty():
        curr_path, action_sequence = open_q.get()
        last_state = curr_path[-1]

        # stop if we're at the end state
        if last_state.is_end():
            continue

        for action in game.get_actions(last_state):
            succ, reward = get_next_state_from_world_model(last_state, action, game, world_model)
            if reward > best_reward:
                best_reward = reward
                best_path = curr_path + [succ]
                best_action_sequence = action_sequence + [action]
            if succ not in seen:
                open_q.put([curr_path + [succ], action_sequence + [action]])
                seen.add(succ)

    return best_path, best_action_sequence, best_reward


def plan_to_goal(game, start_state, world_model):
    """
    Get a list of all state-action pairs
    """
    assert world_model
    open_q = Queue()
    open_q.put([[start_state], []])
    seen = {start_state}

    found_path_to_goal = False
    best_path, best_action_sequence = [], []
    if len(game.get_actions(start_state)) == 0:
        return best_path, best_action_sequence, False
    while not open_q.empty():
        curr_path, action_sequence = open_q.get()
        last_state = curr_path[-1]

        # stop if we're at the end state
        if last_state.is_end():
            continue

        for action in game.get_actions(last_state):
            succ, reward = get_next_state_from_world_model(last_state, action, game, world_model)
            if succ.is_goal_state():
                best_path = curr_path + [succ]
                best_action_sequence = action_sequence + [action]
                found_path_to_goal = True
                break
            if succ not in seen:
                open_q.put([curr_path + [succ], action_sequence + [action]])
                seen.add(succ)

        if found_path_to_goal:
            break

    # if we don't find any path to the goal, take a random action
    if not found_path_to_goal:
        chosen_action = np.random.choice(game.get_actions(start_state))
        best_action_sequence = [chosen_action]
        best_path = [start_state, get_next_state_from_world_model(start_state, chosen_action, game, world_model)[0]]

    return best_path, best_action_sequence, found_path_to_goal

def get_psrl_actions(
        game,
        world_model: MemoizedWorldModel,
        start_state = None
):
    assert world_model

    if start_state is None:
        start_state = game.get_start_state()

    # next, get a set of actions that leads to the optimal reward
    return get_optimal_action_sequence(game, start_state, world_model)

game_idx = 3
if __name__ == "__main__":

    from pyprojroot import here
    import dill

    games = dill.load(open(here("data/games/experiment_randomized_episodic_games.p"), "rb"))
    test_game = games[game_idx]
    knowledge = set()
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

    sampler = DSLSampler(knowledge, known_items)
    world_model = MemoizedWorldModel(sampler)

    path, actions, reward = get_psrl_actions(test_game,
                                             world_model
                                             )
    print(path)
    print(actions)
    print(reward)

    state = test_game.get_start_state()
    total_reward = 0
    for action in actions:
        if action == 0:
            state, reward = test_game.take_action(state, action)
        else:
            state, reward = get_next_state_from_world_model(
                state,
                action,
                test_game,
                world_model=world_model
            )

        total_reward += reward
