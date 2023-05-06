"""
A model for the episodic task
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Iterable
from model import DSLLogicalAgent
from psrl import plan_to_goal, MemoizedWorldModel

class EpisodicLogicalAgent(DSLLogicalAgent):
    """
    The DSL agent, but it does multiple episodes before transmitting
    """

    def __init__(
            self,
            channel_capacity: Optional[int] = None,
            can_abstract: bool = True,
            message_objective: str = "divergence",
            message_selection_method: str = "decremental",
            init_possible_items: Iterable[dict] = (),
            use_cuda: bool = True,
            n_episodes: int = 3
    ):
        super().__init__(
            channel_capacity,
            can_abstract,
            message_objective,
            message_selection_method,
            init_possible_items,
            use_cuda
        )
        self.n_episodes = n_episodes

    def start_episode(self, game, start_state):
        """Update possible items and draw a new posterior sample"""
        self.update_possible_items(start_state)
        self.sampler = self.get_sampler()
        self.memoized_world_model = MemoizedWorldModel(self.sampler)

    def compute_policy(self, game, state):
        """Do PSRL to get a policy"""
        world_model = self.get_memoized_world_model()
        _, actions, _ = plan_to_goal(
            game,
            start_state=state,
            world_model=world_model,
        )
        self.psrl_actions = actions

    def solution_policy(self, game, state):
        """Return the action to take in a particular state"""
        if self.psrl_actions is None or len(self.psrl_actions) == 0:
            self.compute_policy(game, state)

        return self.psrl_actions.pop(0)

    def solve_task(self, game) -> int:
        """
        Solve the task given the current knowledge and solution policy
        """
        #FIXME: at the moment can't run this in isolation because reset
        # is needed to reset the memoized world model. called in full_loop.
        state = game.get_start_state()
        self.start_episode(game, state)
        while not state.is_end():
            action = self.solution_policy(game, state)
            # print(f"*** Taking action {action} ***")
            new_state, _ = game.take_action(state, action)
            self.update_knowledge(game, state, action, new_state)
            state = new_state

        score = 1 if state.is_goal_state() else 0

        return score

    def demonstrate(self, game, test_knowledge, n=10):
        """
        Complete n episodes of the task without learning anything and return
        the mean score
        """
        scores = []
        for i in range(n):
            # re-receive the knowledge (to resample the world model)
            self.reset()
            self.receive_message(test_knowledge)
            for ep in range(self.n_episodes):
                state = game.get_start_state()
                while not state.is_end():
                    self.compute_policy(game, state)
                    action = self.solution_policy(game, state)
                    new_state, reward = game.take_action(state, action)
                    state = new_state
                score = 1 if state.is_goal_state() else 0
                scores.append(score)

        return np.mean(scores)

    def full_loop(self, task, message = (), transmit: bool = True):
        """
        Receive a message, complete multiple episodes of a game,
        then transmit a message
        """
        self.reset() #reset first, so memoized world model is reset
        self.receive_message(message)
        scores = []
        for episode in range(self.n_episodes):
            score = self.solve_task(task)
            scores.append(score)
        if transmit:
            new_message = self.transmit_message(
                task,
                channel_capacity=self.channel_capacity
            )
        else:
            new_message = None

        return scores, new_message

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

game_idx = 1
n_chains = 100
chain_length = 10
n_episodes = 5
channel_capacities = (None, 1, 3, 5, 7, 10)
abstraction_settings = (True, False)
message_selection_method = "anti_unification"
if __name__ == "__main__":

    from pyprojroot import here
    import dill

    games = dill.load(open(here("data/games/experiment_randomized_episodic_games.p"), "rb"))

    game = games[game_idx]

    chain_scores = []
    for chain_idx in tqdm(range(n_chains)):
        for abstraction_setting in abstraction_settings:
            for channel_capacity in channel_capacities:
                curr_msg = set()

                # abstraction doesn't make sense if the channel is unlimited
                if channel_capacity is None and abstraction_setting:
                    continue

                for pos in range(chain_length):

                    received_message_chars = sum([len(str(x)) for x in curr_msg])
                    received_message_n_items = len(curr_msg)
                    received_message_n_abstract_items = len([x for x in curr_msg if x.is_abstract()])
                    received_message = ", ".join([str(x) for x in curr_msg])

                    model = EpisodicLogicalAgent(
                        channel_capacity=channel_capacity,
                        can_abstract=abstraction_setting,
                        init_possible_items=all_possible_items,
                        n_episodes=n_episodes,
                        message_selection_method=message_selection_method
                    )
                    scores, curr_msg = model.full_loop(
                        game,
                        message = curr_msg,
                        transmit = pos != chain_length - 1  # don't send a message if we are at the end of the chain
                    )

                    for ep_num, score in enumerate(scores):
                        chain_scores.append({
                            "chain_idx": chain_idx,
                            "channel_capacity": channel_capacity,
                            "can_abstract": int(abstraction_setting),
                            "chain_pos": pos,
                            "episode": ep_num,
                            "score": score,
                            "received_msg": received_message,
                            "received_msg_chars": received_message_chars,
                            "received_msg_n_items": received_message_n_items,
                            "received_msg_n_abstract_items": received_message_n_abstract_items,
                            "knowledge_size": len(model.knowledge)
                        })

    pd.DataFrame(chain_scores).to_csv(here(f"data/simulated_randomized_episodic_scores_game={game_idx}.csv"), index=False)
