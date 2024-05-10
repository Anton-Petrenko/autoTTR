# AutoTTR
This repository is the codebase for a proposed automated Ticket to Ride agent to be trained using no historical data.
All architectures are directly taken from AlphaGo Zero and adjusted to accomodate the complexities of Ticket to Ride [^1]. Much of this document is a synopsis of AlphaGo Zero.

## Network Architecture
![AutoTTR Network Architecture](https://i.imgur.com/Hb0PGU3.png)

+ **Input Vector** - a combined vector of all relevant information from a given Ticket to Ride state
  - Available destinations (one hot)
  - Destinations held by current player (one hot)
  - Destination counts per opponent (integer encoded)
  - Routes taken by player (one hot)
  - Available colors (integer encoded)
  - Colors held by opponents (integer encoded & 'fog of war')
+ **Outputs** - these outputs are designed to be used to calculate moves the network would like to make
  1. Action to take
       + Place Route
       + Draw Face Up Card
       + Draw Face Down Card
       + Draw Destination Card
  2. Importance of each color
  3. Importance of each destination card (by index of possible card pickup - NOT for each destination card)
  4. Importance of each route
  5. Probability of winning from this state

## Training Architecture
![AutoTTR Training](https://i.imgur.com/5tU1ZwM.png)

To generate the training data for the neural network, we ask the network to play a game of Ticket to Ride against itself.

For each state encountered by the neural network, a Monte Carlo Search is performed. This search uses the neural network to expand and select nodes.

After the Monte Carlo Search has completed, one action from the root node is selected and performed in the original game.

This process repeats until the game is over - a new game is then started. 

Game are continuously played and histories are stored. This is the training data.

## Monte Carlo Search Algorithm

The specific algorithm of Monte Carlo Search is in the AlphaGo Zero paper and mirrored directly in AutoTTR. 

It is not fully documented here yet.

## Last Checkpoint

Creating labels for the neural network in network.py

## TODO

- [ ] **Game Engine**
  - [ ] Make sure the player who initiated the last turn actually gets a last turn
  - [ ] Test gameWinner() function
  - [ ] More specific logs? Terminal state logs and who caused the end of the game and such
  - [ ] color counting for each player
  - [ ] reshuffle train cards facing up based on game rule conditions
  - [ ] (self.turn - 1) % len(self.players) the way to get the right person to go from self.players?
- [ ] **Monte Carlo Search**
  - [ ] in MCTS, do terminal states in the simulation have different things to backprop? consult AGZero paper
  - [ ] Do typecasting and object creating for the bucket in training.py
  - [ ] Learning rate schedule - how should it be adjusted for this implementation? (training.py)
- [ ] **Network**
  - [ ] getting logits for an action (is multiplying the best way?)
  - [ ] test network output to logit conversion!
  - [ ] verify that game translation to network input is valid

[^1]: Silver, David, et al. “Mastering the Game of Go without Human Knowledge.” 
Nature, vol. 550, no. 7676, Oct. 2017, pp. 354–359
