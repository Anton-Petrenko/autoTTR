'''
benchmark.py
---------
Logic is included here to run clusters of games with various settings
'''

import random
from engine.game import Game, Player
import matplotlib.pyplot as plt
from network.network import AutoTTR
from network.training import getLogitMoveExternal

class Benchmark:

    def __init__(self, gameSizes: int, players: list[int], minGamesPerPlayer: int):
        """
        gameSizes > num of players in each game simulation

        players > pool of players to choose from by number where positive integer is a network save number in saved folder & -1 is a random opponent

        A game with only 4 players with 1 minGamePerPlayer will result in one simulated game whose logs will be recorded
        """
        
        gamesNeeded = len(players) * minGamesPerPlayer // 4
        self.players = players
        self.playerCache = {}
        self.gamesCount = dict()
        self.pointsCount = {}
        self.averages = {}

        while len(self.players) >= 4:
            playersList = random.sample(self.players, 4)

            for player in playersList:
                if player not in self.gamesCount:
                    self.gamesCount[player] = 0
                    self.pointsCount[player] = 0
            if all(self.gamesCount[player] < minGamesPerPlayer for player in playersList):
                print(f"Running {playersList}...")
                playing = [Player(f"benchmark{playersList[0]}"), Player(f"benchmark{playersList[1]}"), Player(f"benchmark{playersList[2]}"), Player(f"benchmark{playersList[3]}")]
                
                game = None
                if len(players) == 4 and minGamesPerPlayer == 1:
                    game = Game(playing, "USA", True, False, False)
                else:
                    game = Game(playing, "USA", False, False, False)

                while not game.gameIsOver:
                    if game.turn % 25 == 0:
                        print(f"\tTurn {game.turn}...")
                    validmoves = game.getValidMoves()
                    if len(validmoves) == 0:
                        continue
                    self.handleMove(validmoves, game)
                
                for player in game.finalStandings:
                    print(f"\t{player.name} | {player.points} points")
                    playerID = int(player.name[9:])

                    if playerID not in self.gamesCount:
                        self.gamesCount[playerID] = 1
                        self.pointsCount[playerID] = player.points
                    else:
                        self.gamesCount[playerID] += 1
                        self.pointsCount[playerID] += player.points

                    if self.gamesCount[playerID] >= minGamesPerPlayer:
                        print(f"Removing {playerID} from {self.players}")
                        self.players.remove(playerID)
    
        for playerID, numGames in self.gamesCount.items():
            self.averages[playerID] = self.pointsCount[playerID] / numGames
        

        self.averages = dict(sorted(self.averages.items()))
        categories = list(str(data) for data in self.averages.keys())
        values = list(self.averages.values())

        plt.bar(categories, values, color='blue', alpha=0.7)

        plt.xlabel("Model")
        plt.ylabel("Average Score")
        plt.title("Average Scores per Model")

        plt.show()
    
    def handleMove(self, validmoves: list, game: Game):
        playerID = game.players[game.turn % len(game.players)].name[9:]
        playerID = int(playerID)

        # Random
        if playerID == -1:
            action = random.choice(validmoves)
            game.play(action)
        elif playerID >= 0:
            
            model: AutoTTR = None

            if playerID not in self.playerCache:
                print(f"Initializing {playerID}...")
                model = AutoTTR(0.01, 0, 0, playerID)
                self.playerCache[playerID] = model
            else:
                model = self.playerCache[playerID]
            
            maxProb = -1
            action = None
            outModel = model.think(game)
            for move in validmoves:
                prob = getLogitMoveExternal(outModel, move)
                if prob > maxProb:
                    maxProb = prob
                    action = move
            game.play(action)
