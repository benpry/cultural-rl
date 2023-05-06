# Cultural reinforcement learning

This repo contains code and data for the paper "Cultural reinforcement learning: a framework for modeling cumulative culture on a limited channel" by Prystawski, Arumugam, and Goodman, presented at CogSci 2023.

It is organized into three sub-folders, `code`, `data`, and `figures`.

## Generating games

You can generate games to run the model on using `code/generate_games.py`. this file relies on two other files: `game.py` contains the implementation of the games and `game_generation_utils.py` contains helper functions for creating games. By default, it creates two types of anti-unifiable games: those where only the colors of the inputs matter and those where only the shapes matter. It generates 50 per type, but this can be changed by modifying the constant `n_games_per_type`. Generating games produces the output file `data/games/experiment_randomized_episodic_games.p`

## Running the model

We run the model on the generated games with `code/run_model_experimental_games.py`. This file contains several constants which you can modify to try different conditions, like the channel capacities and whether abstraction is possible. It relies on `model.py` and `episodic_model.py`, which contain the base model code, `recipe_learning.py` which handles the DSL statements, and `psrl.py` which handles planning.

To run the selective social learning setup, you can run `code/selective_social_learning.py`, which runs populations of agents which choose who to learn from in the previous generation via selective social learning.

## Analyzing data

There are three analysis files, all of which are in Quarto markdown format. `ChainAnalysis.qmd` analyzes scores by chain position and absolute episode number. `MessageAnalysis.qmd` looks at the proportion of abstract statements in a message and correlations between the scores of the sender and receiver of a message. `SSLAnalysis.qmd` compares different selective social learning conditions.

