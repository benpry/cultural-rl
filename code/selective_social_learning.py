"""
Run the agent in populations with selective social learning
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from episodic_model import EpisodicLogicalAgent
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

game_idx = 0
n_simulations = 1000
population_sizes = (10, 100)
chain_length = 10
n_episodes = 5
channel_capacities = (5, 10)
abstraction_settings = (True,)
ssl_settings = ("random", "demonstration", "learning")
message_selection_method = "anti_unification"
if __name__ == "__main__":

    games = dill.load(open(here("data/games/experiment_randomized_episodic_games.p"), "rb"))
    game = games["only_color"][game_idx]

    rows = []
    prev_gen_scores, prev_gen_messages = [], []
    for ssl_setting in ssl_settings:
        for channel_capacity in channel_capacities:
            for abstraction_setting in abstraction_settings:
                for population_size in population_sizes:
                    for sim in range(n_simulations // population_size):
                        for generation in range(chain_length):
                            messages = []
                            scores = []
                            print(f"gen {sim}-{generation}")

                            for i in tqdm(range(population_size)):

                                if generation == 0:
                                    msg = ()
                                else:
                                    # select a message to socially learn from
                                    if ssl_setting != "random":
                                        prev_generation_probs = np.exp(100 * np.array(prev_gen_scores))
                                        msg_idx = np.random.choice(population_size, p=prev_generation_probs / np.sum(prev_generation_probs))
                                    else:
                                        msg_idx = np.random.choice(population_size)
                                    msg = prev_gen_messages[msg_idx]

                                model = EpisodicLogicalAgent(
                                    channel_capacity=channel_capacity,
                                    can_abstract=abstraction_setting,
                                    init_possible_items=all_possible_items,
                                    n_episodes=n_episodes,
                                    message_selection_method=message_selection_method
                                )

                                episode_scores, curr_msg = model.full_loop(
                                    game,
                                    message = msg,
                                    transmit = generation != chain_length - 1
                                )

                                messages.append(curr_msg)

                                if curr_msg is not None and ssl_setting == "demonstration":
                                    demo_model = EpisodicLogicalAgent(
                                        channel_capacity=channel_capacity,
                                        can_abstract=abstraction_setting,
                                        init_possible_items=all_possible_items,
                                        n_episodes=n_episodes,
                                        message_selection_method=message_selection_method
                                    )
                                    scores.append(demo_model.demonstrate(game, model.knowledge, n=10))
                                elif ssl_setting == "learning":
                                    scores.append(np.mean(episode_scores))

                                rows.append({
                                    "ssl_setting": ssl_setting,
                                    "simulation_num": sim,
                                    "agent_num": i,
                                    "population_size": population_size,
                                    "channel_capacity": channel_capacity,
                                    "can_abstract": int(abstraction_setting),
                                    "generation": generation,
                                    "score": np.mean(episode_scores),
                                })

                            prev_gen_messages = messages
                            prev_gen_scores = scores

    pd.DataFrame(rows).to_csv(here("data/ssl_scores.csv"), index=False)
