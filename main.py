from internal.types import Agent
from agents.network import Network
from internal.training import Training

training = Training("USA", [Agent("Spongebob"), Agent("Patrick"), Agent("Sandy"), Agent("Squidward")], Network("BetaStopOne"))
training.train()