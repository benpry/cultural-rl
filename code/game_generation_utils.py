"""
helper functions for automatically generating games
"""
from frozendict import frozendict
import numpy as np
from random import shuffle
from game import Game
from itertools import product
from psrl import plan_to_goal, MemoizedWorldModel, DSLSampler
from episodic_model import EpisodicLogicalAgent
from recipe_learning import Statement

COLORS = ["red", "green", "blue"]
SHAPES = ["triangle", "square", "pentagon"]

def create_recipe_fn():

    color_mapping = {}
    for x in COLORS:
        color_mapping[x] = {}
        for y in COLORS:
            color_mapping[x][y] = np.random.choice(COLORS)

    # this is dumb and should be a for loop, but I'm implementing it quickly
    shape_mapping = {}
    for x in SHAPES:
        shape_mapping[x] = {}
        for y in SHAPES:
            shape_mapping[x][y] = np.random.choice(SHAPES)

    def recipe_fn(i1, i2):
        f1, f2 = i1["features"], i2["features"]
        new_color = color_mapping[f1["color"]][f2["color"]]
        new_shape = shape_mapping[f1["shape"]][f2["shape"]]

        return {
            "features": frozendict({
                "color": new_color,
                "shape": new_shape
            })
        }

    return recipe_fn

def create_recipe_fn_only_color(return_dict=False):

    outputs = [
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
    shuffle(outputs)

    color_mapping = {}
    for i, x in enumerate(COLORS):
        color_mapping[x] = {}
        for j, y in enumerate(COLORS):
            color_mapping[x][y] = outputs[i * len(COLORS) + j]

    def recipe_fn(i1, i2):
        f1, f2 = i1["features"], i2["features"]
        new_item = color_mapping[f1["color"]][f2["color"]]

        return {"features": frozendict(new_item)}

    if return_dict:
        return recipe_fn, color_mapping
    else:
        return recipe_fn

def create_recipe_fn_only_shape(return_dict=False):

    outputs = [
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
    shuffle(outputs)

    shape_mapping = {}
    for i, x in enumerate(SHAPES):
        shape_mapping[x] = {}
        for j, y in enumerate(SHAPES):
            shape_mapping[x][y] = outputs[i * len(SHAPES) + j]

    def recipe_fn(i1, i2):
        f1, f2 = i1["features"], i2["features"]
        new_item = shape_mapping[f1["shape"]][f2["shape"]]

        return {"features": frozendict(new_item)}

    if return_dict:
        return recipe_fn, shape_mapping
    else:
        return recipe_fn

def create_recipe_fn_only_left(return_dict=False):

    outputs = [
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
    shuffle(outputs)

    output_mapping = {}
    for i, x in enumerate(COLORS):
        output_mapping[x] = {}
        for j, y in enumerate(SHAPES):
            output_mapping[x][y] = outputs[i * len(SHAPES) + j]

    def recipe_fn(i1, i2):
        f1 = i1["features"]
        new_item = output_mapping[f1["color"]][f1["shape"]]

        return {"features": frozendict(new_item)}

    if return_dict:
        return recipe_fn, output_mapping
    else:
        return recipe_fn

def create_recipe_fn_no_structure():
    """
    Totally random recipes, nothing abstract to be learned
    """

    all_items = [
        frozendict({"color": "green", "shape": "triangle"}),
        frozendict({"color": "green", "shape": "square"}),
        frozendict({"color": "green", "shape": "pentagon"}),
        frozendict({"color": "red", "shape": "triangle"}),
        frozendict({"color": "red", "shape": "square"}),
        frozendict({"color": "red", "shape": "pentagon"}),
        frozendict({"color": "blue", "shape": "triangle"}),
        frozendict({"color": "blue", "shape": "square"}),
        frozendict({"color": "blue", "shape": "pentagon"})
    ]
    recipes = {}
    for x in all_items:
        recipes[x] = {}
        for y in all_items:
            z = np.random.choice(all_items)
            recipes[x][y] = z

    def recipe_fn(i1, i2):
        f1, f2 = i1["features"], i2["features"]
        new_item = recipes[x][y]

        return {"features": frozendict(new_item)}

    return recipe_fn

def create_value_fn():

    color_values = {}
    for color in COLORS:
        color_values[color] = np.round(np.random.uniform(0, 100))
    shape_values = {}
    for shape in SHAPES:
        shape_values[shape] = np.round(np.random.uniform(0, 100))

    def value_fn(i):
        f = i["features"]
        return color_values[f["color"]] + shape_values[f["shape"]]

    return value_fn

def create_starting_inventory(n_items):

    existing_inventory = []

    for i in range(n_items):
        item_features = {"color": np.random.choice(COLORS), "shape": np.random.choice(SHAPES)}
        already_exists = False
        for existing_item in existing_inventory:
            if existing_item["features"] == item_features:
                already_exists = True
                existing_item["n"] += 1
                break
        if not already_exists:
            existing_inventory.append({"n": 1, "features": item_features})

    starting_inventory = [frozendict({"n": item["n"], "features": frozendict(item["features"])})
                          for item in existing_inventory]

    return starting_inventory

def create_starting_inventory_fixed_size(n_unique, n_items):

    existing_inventory = []
    for i in range(n_items):

        if len(existing_inventory) == n_unique:
            item_features = existing_inventory[np.random.choice(len(existing_inventory))]["features"]
        else:
            item_features = {"color": np.random.choice(COLORS), "shape": np.random.choice(SHAPES)}
        already_exists = False
        for existing_item in existing_inventory:
            if existing_item["features"] == item_features:
                already_exists = True
                existing_item["n"] += 1
                break
        if not already_exists:
            existing_inventory.append({"n": 1, "features": item_features})

    starting_inventory = [frozendict({"n": item["n"], "features": frozendict(item["features"])})
                          for item in existing_inventory]

    return starting_inventory

def is_achievable(game):
    """
    check whether it is possible, given perfct knowledge about a game,
    to achieve every possible goal. Episodic games only.
    """

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

        z = game.recipe_fn(x, y)["features"]
        all_statements.add(
            Statement([x["features"], y["features"]], z)
        )

    model = EpisodicLogicalAgent()

    model.knowledge = all_statements
    model.update_possible_items()

    sampler = DSLSampler(model.knowledge, model.possible_items)
    world_model = MemoizedWorldModel(sampler)

    for goal in game.possible_goals:
        start_state = game.get_start_state()
        start_state.goal = goal

        states, actions, reward = plan_to_goal(
            game,
            start_state,
            world_model
        )

        if reward is False:
            return False

    return True

def is_achievable_stochastic(game, n_evals=50):
    """Test if a game with a random start state is achievable a bunch of times"""
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

        z = game.recipe_fn(x, y)["features"]
        all_statements.add(
            Statement([x["features"], y["features"]], z)
        )

    model = EpisodicLogicalAgent()

    model.knowledge = all_statements
    model.update_possible_items()

    sampler = DSLSampler(model.knowledge, model.possible_items)
    world_model = MemoizedWorldModel(sampler)

    for i in range(n_evals):
        start_state = game.get_start_state()

        states, actions, reward = plan_to_goal(
            game,
            start_state,
            world_model
        )

        if reward is False:
            return False

    return True
