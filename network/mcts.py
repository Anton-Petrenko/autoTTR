import time
import os
import multiprocessing
from random import choice
from typing import Literal
from engine.game import *

class Node:
        def __init__(self, visits):
            self.visits = visits
            self.children: dict[Action, Node] = {}

class FlatMonteCarlo:
    """
    The class encompassing FlatMonteCarlo search given a game state
    """

    def __init__(self, game: Game, seconds_per_branch: int, end_goal: Literal['wins', 'points']):
        assert end_goal in ['wins', 'points']
        self.game = game
        self.end_goal = end_goal
        self.seconds_per_branch = seconds_per_branch

    def simulate(self, action: Action) -> tuple[int, int]:
        assert isinstance(action, Action)
        wins = 0
        start_time = time.time()
        player_id = self.game.turn % len(self.game.players)
        total_simulations = 0

        while time.time() - start_time < self.seconds_per_branch:
            game_clone = self.game.clone()
            game_clone.play(action)
            cur_node = self.root.children[action]
            cur_node.visits += 1

            while not game_clone.gameIsOver:
                valid_moves = game_clone.getValidMoves()
                if len(valid_moves) == 0:
                    continue
                chosen_action = choice(valid_moves)
                if chosen_action in cur_node.children.keys():
                    cur_node = cur_node.children[chosen_action]
                    cur_node.visits += 1
                else:
                    cur_node.children[chosen_action] = Node(0)
                    cur_node = cur_node.children[chosen_action]
                    cur_node.visits += 1

                game_clone.play(chosen_action)
            
            total_simulations += 1

            if self.end_goal == 'points':
                wins += game_clone.players[player_id].points
            elif self.end_goal == 'wins':
                if game_clone.finalStandings[0].turnOrder == player_id:
                    wins += 1
            else:
                raise ValueError("Value error in simulate within flatmontecarlo")
        
        # print(f"{os.getpid()} Flat MCTS Simulation ran {total_simulations} times for given action.")
        return wins, total_simulations

    def search(self) -> Action:
        """Returns the Action to take as a result of a flat monte carlo search"""

        self.root = Node(0)
        valid_moves = self.game.getValidMoves()

        for action in valid_moves:
            self.root.children[action] = Node(0)

        actions = list(self.root.children.keys())

        # print(f"{os.getpid()} Starting parallel processes...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.simulate, actions)
        print(results)
        max_ratio = None
        best_action = None
        total_simulations = sum([x[1] for x in results])
        # print(f"{os.getpid()} Flat MCTS ran {total_simulations} times after all processes finished.")
        for _action, wins in list(zip(actions, [x[0] for x in results])):
            ratio = wins / total_simulations
            if max_ratio == None:
                max_ratio = ratio
                best_action = _action
            elif wins / total_simulations > max_ratio:
                max_ratio = ratio
                best_action = _action

        assert isinstance(best_action, Action)
        return best_action

    def evaluate(self):
        pass
