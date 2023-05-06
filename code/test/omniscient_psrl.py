"""
A file for debugging PSRL
"""
import dill
from pyprojroot import here
from itertools import product
from recipe_learning import Statement
from model import DSLLogicalAgent
from psrl import get_psrl_actions, MemoizedWorldModel, DSLSampler


game_idx = 0
use_no_sell_games = True
if __name__ == "__main__":

    path_str = "data/games/experiment_randomized_episodic_games.p"
    games = dill.load(open(here(path_str), "rb"))

    test_game = games[game_idx]

    if use_no_sell_games and False:
        all_features = {
            "color": ["green", "yellow", "red", "blue"],
            "shape": ["triangle", "square", "pentagon", "hexagon"]
        }
    else:
        all_features = {
            "color": ["green", "red", "blue"],
            "shape": ["triangle", "square", "pentagon"]
        }

    # compile all the knowledge
    # iterate over possible recipes
    all_statements = set()
    for possible_recipe in product(all_features["color"], all_features["shape"], repeat=2):
        x = {"features": {"color": possible_recipe[0], "shape": possible_recipe[1]}}
        y = {"features": {"color": possible_recipe[2], "shape": possible_recipe[3]}}

        z = test_game.recipe_fn(x, y)["features"]
        all_statements.add(
            Statement([x["features"], y["features"]], z)
        )

    for x in all_statements:
        print(str(x))

    model = DSLLogicalAgent()
    model.knowledge = all_statements
    model.update_possible_items()

    sampler = DSLSampler(model.knowledge, model.possible_items)
    world_model = MemoizedWorldModel(sampler)

    states, actions, reward = get_psrl_actions(
        test_game,
        world_model
    )

    print(states)
    print(actions)
    print(reward)

    state = test_game.get_start_state()
    print(state)
    total_reward = 0
    for action in actions:
        print(action)
        state, reward = test_game.take_action(state, action)
        print(state)
        total_reward += reward

    print(total_reward)
