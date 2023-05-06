"""
This file runs the episodic agent on experimental tasks and saves statistics about its performance
"""
from episodic_model import EpisodicLogicalAgent
import pandas as pd
from tqdm import tqdm
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

games_file = "data/games/experiment_randomized_episodic_games.p"
games_to_run = {
    "only_color": range(50),
    "only_shape": range(50),
}
n_chains = 100
chain_length = 10
n_episodes = 5
channel_capacities = (None, 1, 3, 5, 7, 10)
abstraction_settings = (True, False)
message_selection_method = "anti_unification"
if __name__ == "__main__":

    with open(here(games_file), "rb") as fp:
        all_games = dill.load(fp)

    chain_scores = []
    for game_type in games_to_run.keys():
        print(game_type)
        for game_idx in tqdm(games_to_run[game_type]):
            game = all_games[game_type][game_idx]
            for chain_idx in range(n_chains):
                for channel_capacity in channel_capacities:
                    for abs_setting in abstraction_settings:

                        if channel_capacity is None and abs_setting is False:
                            continue

                        curr_msg = set()
                        for pos in range(chain_length):

                            received_message_chars = sum([len(str(x)) for x in curr_msg])
                            received_message_n_items = len(curr_msg)
                            received_message_n_abstract_items = len([x for x in curr_msg if x.is_abstract()])
                            received_message = ", ".join([str(x) for x in curr_msg])

                            model = EpisodicLogicalAgent(
                                channel_capacity=channel_capacity,
                                can_abstract=abs_setting,
                                init_possible_items=all_possible_items,
                                n_episodes=n_episodes,
                                message_selection_method=message_selection_method
                            )
                            scores, curr_msg = model.full_loop(
                                game,
                                message = curr_msg,
                                transmit = True
                            )

                            for ep_num, score in enumerate(scores):
                                chain_scores.append({
                                    "game_type": game_type,
                                    "game_idx": game_idx,
                                    "chain_idx": chain_idx,
                                    "channel_capacity": channel_capacity,
                                    "can_abstract": int(abs_setting),
                                    "chain_pos": pos,
                                    "episode": ep_num,
                                    "score": score,
                                    "received_msg": received_message,
                                    "received_msg_chars": received_message_chars,
                                    "received_msg_n_items": received_message_n_items,
                                    "received_msg_n_abstract_items": received_message_n_abstract_items,
                                    "knowledge_size": len(model.knowledge)
                                })


    pd.DataFrame(chain_scores).to_csv(here(f"data/model_data_all_experimental_tasks.csv"), index=False)
