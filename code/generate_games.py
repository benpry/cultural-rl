"""
Generate games for the human experiment
"""
from game_generation_utils import create_starting_inventory_fixed_size, create_recipe_fn_only_color, create_recipe_fn_only_shape, create_recipe_fn_only_left, is_achievable_stochastic
from game import RandomEpisodicGame
from pyprojroot import here
import dill

all_possible_items = [
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

def generate_game(recipe_fn_generator):
    recipe_fn, recipe_dict = recipe_fn_generator(return_dict=True)

    return RandomEpisodicGame(
        recipe_fn=recipe_fn,
        possible_items=all_possible_items
    ), recipe_dict

recipe_types = {
    "only_color": create_recipe_fn_only_color,
    "only_shape": create_recipe_fn_only_shape
}
n_games_per_type = 50
if __name__ == "__main__":

    all_games = {}
    all_recipe_dicts = {}
    for recipe_type, create_recipe_fn in recipe_types.items():
        print(recipe_type)
        games = []
        recipe_dicts = []
        while len(games) < n_games_per_type:
            game, recipe_dict = generate_game(create_recipe_fn)
            while not is_achievable_stochastic(game):
                game, recipe_dict = generate_game(create_recipe_fn)
            games.append(game)
            recipe_dicts.append(recipe_dict)
            print(len(games))

        all_games[recipe_type] = games
        all_recipe_dicts[recipe_type] = recipe_dicts

    dill.dump(all_games, open(here("data/games/experiment_randomized_episodic_games.p"), "wb"))
    dill.dump(all_recipe_dicts, open(here("data/games/experiment_randomized_recipe_dicts.p"), "wb"))
