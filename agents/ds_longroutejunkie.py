import networkx as nx
from engine.game import Game

class LongRouteJunkie:
    def __init__(self):
        self.current_objective_route = None
        self.current_objective_route = None
        self.players_previous_points = -1
    
    def decide(self, game: Game):
        possible_moves = game.getValidMoves()
        if len(possible_moves) == 0:
            print("No possible moves")
            return None
        
        if possible_moves[0].action == 3 and possible_moves[0].askingForDeal == False:
            self.players_previous_points = -1
            print("ds_longroutejunkie.py 18 - does this ever even return true?")
            quit()
        
        claim_route_moves = []
        draw_train_card_moves = []

        for move in possible_moves:
            if move.action == 0:
                claim_route_moves.append(move)
            elif move.action == 1 or move.action == 2:
                draw_train_card_moves.append(move)
        
        total_current_points = 0
        for i in range(0, len(game.players)):
            total_current_points += game.players[i].points
        
        if self.players_previous_points < total_current_points:
            # x = self.generate_game_plan(game)
            pass
    
    def chooseDestinations(self, moves, game, min_weight_edge):
        best_move = (0, None)
        least_worst_move = (0, None)
        joint_graph = self.joint_graph(game, min_weight_edge)

        for m in moves:
            current_move_value = 0
            number_of_trains_needed = 0
            points = 0
            for destination in m:
                temp = self.calculate_value()

    def calculate_value(self, destination, points, ):
        pass
    
    def joint_graph(self, game: Game, min_weight_edge):
        free_connections_graph = self.free_routes_graph(game.board, len(game.players), min_weight_edge)
        player_edges = [edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.turn % len(game.players)]

        joint_graph = free_connections_graph
        for edge in player_edges:
            joint_graph.add_edge(edge[0], edge[1], weight=0, color='none', owner=game.turn % len(game.players))
    
        return joint_graph

    def free_routes_graph(self, graph, number_of_players, min_weight_edge=0):
        G = nx.MultiGraph()

        visited_nodes = []

        for node1 in graph:
            for node2 in graph:
                if node2 not in visited_nodes:
                    #some code skipped - will always be 4 player

                    for edge in graph[node1][node2]:
                        if graph[node1][node2][edge]['owner'] == None and graph[node1][node2][edge]['weight'] >= min_weight_edge:
                            G.add_edge(node1, node2, weight=graph[node1][node2][edge]['weight'], color=graph[node1][node2][edge]['color'], owner=-1)
            
            visited_nodes.append(node1)
        
        return G