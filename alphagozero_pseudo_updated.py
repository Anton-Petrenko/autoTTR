"""Pseudocode description of the AlphaZero algorithm."""

from __future__ import google_type_annotations
from __future__ import division

import math
import numpy
import tensorflow as tf
from typing import List

##########################
####### Helpers ##########


class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000 # ???

    self.num_sampling_moves = 30 
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0 # N(s, a)
    self.to_play = -1
    self.prior = prior   # P(s, a)
    self.value_sum = 0   # W(s, a)
    self.children = {}

  def expanded(self):
    """
    Returns if the current game node has been expanded or not
    """
    return len(self.children) > 0

  def value(self):
    """
    Get the mean action value, Q(s, a) of the node.
    """
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class Game(object):

  def __init__(self, history=None):
    self.history = history or []
    self.child_visits = []
    # an array where, for each move in the game, an array containing a value for each possible action in any state of go is stored, where
    # each action is given a softmax value based on the number of times that move was visited in the mcts tree from that move.
    self.num_actions = 362  # action space size for chess; 11259 for shogi, 362 for Go

  def terminal(self):
    # Game specific termination rules.
    pass

  def terminal_value(self, to_play):
    # Game specific value.
    pass

  def legal_actions(self):
    # Game specific calculation of legal actions.
    return []

  def clone(self):
    return Game(list(self.history))

  def apply(self, action):
    self.history.append(action)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.itervalues()) # how many times the node/root was travelled down the mcts tree
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])
    # add a list to the self object list, where for each possible action in any game of go, if the action is part of the children of the...
    # ... root node, set the value in the list to the visit count / sum_visits, otherwise it was never visited, and the value should be set to 0

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int):
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])
  

  def to_play(self):
    return len(self.history) % 2


class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    # self.buffer is a list of completed games that have been played by the network and simulated in each move by MCTS (max size 100000 games)
    move_sum = float(sum(len(g.history) for g in self.buffer)) # get total number of moves throughout the whole sample to train on
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer]) # select, from the 100000 games played, 4096 (self.batch_size)...
        # ...will be chosen from the sample of games with probabilities assigned to each game. each game is assigned a probability
        # of (moves played in this game / total moves between all games)
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games] # take the sample and create a tuple for each game...
    # ... where the tuple is (game, random integer representing a position in the game history) and put all tuples in a list
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos] # return a list of tuples where each tuple is the...
    # ... (input for the network, tuple 2) where tuple 2 is actually just the labels for all heads of the network


class Network(object):

  def inference(self, image):
    return (-1, {})  # Value, Policy

  def get_weights(self):
    # Returns the weights of this network.
    return []


class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())]
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage() # Passing networks to MCTS
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game) # Since this is done in parallel, games are constantly being played
    # by the network, this step is to store games played into the shared buffer, which is then sampled to train the network
    # the buffer stores games, the games store all past moves


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root) # not sure if this is even used in the long run?
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
  root = Node(0)
  evaluate(root, game, network) ## send input through NN
  add_exploration_noise(config, root) ## adding noise to mcts for exploration, adds 

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.iteritems())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / # Cpuct from paper
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1) # Cpuct * Right Side of U(s, a) equation

  prior_score = pb_c * child.prior # P(s, a) * everything else = U(s, a)
  value_score = child.value() # Getting Q(s, a)
  return prior_score + value_score # Q(s, a) + U(s, a)


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  value, policy_logits = network.inference(game.make_image(-1)) # Obtains (p, v)

  # Expand the node.
  node.to_play = game.to_play() # Set the nodes "to_play" value to reflect whether this player is making the next move
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()} # Map every legal action to the value from the neural net squared
  policy_sum = sum(policy.itervalues()) # Sum all values from dict above (sum the squares)
  for action, p in policy.iteritems(): 
    # for each action: nn value^2 in the dictionary, add children to the given node and set their prior probabilities
    node.children[action] = Node(p / policy_sum)
  return value # return the winning probability only


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.compat.v1.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.compat.v1.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.compat.v1.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=tf.stop_gradient(target_policy)))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()
