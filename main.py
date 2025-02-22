'''
main.py
----------
All scripting should be done here
'''

# TODO: configure default players list when intializing the game object
# TODO: make sure lastAction in game object is only for sequential moves by the same player. (not literally the last move made)
# TODO: currentAction variable in game object - needed?
# TODO: sub len(self.players) for static variable?
# TODO: change checkFaceUp to checkDecks or something - discards may be populated with all other decks exhausted
# NOTE: route checking for 2, 3 player games is done in placeRoute function in Game object
# TODO: check backpropagation 1 - value for nodes that are not the current players nodes
# NOTE !!: To play a move in the engine, you MUST call getValidMoves first and choose from those options. This function has state-altering code in it in certain circumstances
# NOTE: Up until save 199, the model was training on 10 moves per 2 games simulated, this was changed to 50 moves per 2 games simulated
# NOTE: Up until save 255, the model was doing one MCTS simulation per move, this was changed to 100, and network save interval was changed to 5


if __name__ == "__main__":
    # from network.training import NetworkTrainer
    # trainer = NetworkTrainer(numPlayers=4, logs=False, gameSimsPerBatch=2, batchSize=50, trainingSteps=10, networkSaveInterval=5, loadFrom=263, mctsSimsPerMove=100)
    # trainer.run()

    # from engine.benchmark import Benchmark
    # Benchmark(4, [256, 259, 262, -1], 5)

    from engine.game import Player, Game
    from agents.ds_hungry import HungryAgent
    from agents.ds_longroutejunkie import LongRouteJunkie
    players = [Player("Spongebob"), Player("Patrick"), Player("Squidward"), Player("Sandy")]
    game = Game(players, logs=True, visualize=True)
    hungry = LongRouteJunkie()
    while not game.gameIsOver:
        print(game.turn)
        decision = hungry.decide(game)
        if decision == None:
            continue
        game.play(decision)
    # print(hungry.routes_by_color)
    # print(hungry.colors_needed)
    game.draw()