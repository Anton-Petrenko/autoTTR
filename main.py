'''
main.py
----------
All scripting should be done here
'''

# TODO: configure default players list when intializing the game object
# TODO: colorCounts
# TODO: can a terminal state in Ticket to Ride be determined by how many moves have been made? ex. chess is 512
# TODO: Verify network.stateToInput
# TODO: make sure lastAction in game object is only for sequential moves by the same player. (not literally the last move made)
# TODO: currentAction variable in game object - needed?
# TODO: sub len(self.players) for static variable?
# TODO: change checkFaceUp to checkDecks or something - discards may be populated with all other decks exhausted
# NOTE: route checking for 2, 3 player games is done in placeRoute function in Game object
# TODO: check backpropagation 1 - value for nodes that are not the current players nodes
# NOTE !!: To play a move in the engine, you MUST call getValidMoves first and choose from those options. This function has state-altering code in it in certain circumstances

# Trainer
train = False
if train:
    from network.training import NetworkTrainer
    from random import randint
    trainer = NetworkTrainer(numPlayers=4, logs=False, gameSimsPerBatch=2, batchSize=10, trainingSteps=10, networkSaveInterval=10, loadFrom=55)
    trainer.run()

# Tester
test = True
if test:
    model1 = int(input("First model number: "))
    model2 = int(input("Second model number: "))
    
    from network.network import AutoTTR
    from engine.game import Game, Player
    from random import randint
    from network.training import getLogitMoveExternal
    
    model1 = AutoTTR(0.01, 0, 0, model1)
    model2 = AutoTTR(0.01, 0, 0, model2)

    for i in range(1):

        game = Game([Player("model2"), Player("random"), Player("model1"), Player("random")], logs=True)

        while not game.gameIsOver:
            print(game.turn) 
            moves = game.getValidMoves()
            if len(moves) == 0:
                continue
            if game.turn % len(game.players) in [1, 3]:
                # maxProb = -1
                # action = None
                # outModel1 = model1.think(game)
                # for move in moves:
                #     prob = getLogitMoveExternal(outModel1, move)
                #     if prob > maxProb:
                #         maxProb = prob
                #         action = move
                action = moves[randint(0, len(moves)-1)]
                game.play(action)
            elif game.turn % len(game.players) in [0]:
                maxProb = -1
                action = None
                outModel2 = model2.think(game)
                for move in moves:
                    prob = getLogitMoveExternal(outModel2, move)
                    if prob > maxProb:
                        maxProb = prob
                        action = move
                game.play(action)
            elif game.turn % len(game.players) in [2]:
                maxProb = -1
                action = None
                outModel1 = model1.think(game)
                for move in moves:
                    prob = getLogitMoveExternal(outModel1, move)
                    if prob > maxProb:
                        maxProb = prob
                        action = move
                game.play(action)

        for player in game.finalStandings:
            print(f"{player.name} {player.points}")