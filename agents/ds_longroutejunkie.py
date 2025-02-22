import networkx as nx
from engine.game import Game, Action
from engine.objects import color_indexing
import operator
import random

class LongRouteJunkie:
    def __init__(self):
        self.current_objective_route = None
        self.current_objective_color = None
        self.players_previous_points = -1
    
    def decide(self, game: Game):
        possible_moves = game.getValidMoves()
        if len(possible_moves) == 0:
            print("ds_longroutejunkie.py -- No Possible Moves")
            return None

        if possible_moves[0].action == 3 and possible_moves[0].askingForDeal == False:
            self.players_previous_points = -1
            return self.chooseDestinations(possible_moves, game, 3)
        
        claim_route_moves: list[Action] = []
        draw_train_card_moves: list[Action] = []

        for move in possible_moves:
            if move.action == 0:
                claim_route_moves.append(move)
            elif move.action == 1 or move.action == 2:
                draw_train_card_moves.append(move)
        
        total_current_points = 0
        for i in range(0, len(game.players)):
            total_current_points += game.players[i].points
        
        if self.players_previous_points < total_current_points:
            x = self.generate_game_plan(game)
            self.current_objective_route = [x[0], x[1]]
            self.current_objective_color = x[2]
            self.players_previous_points = total_current_points
        
        if self.current_objective_color != None:

            for move in claim_route_moves:
                if (move.route.city1 in self.current_objective_route and move.route.city2 in self.current_objective_route):
                    return move
            
            draw_top_move = None
            draw_wild_move = None
            if self.current_objective_color != 'GRAY':
                for move in draw_train_card_moves:
                    if move.action == 1 and move.colorToDraw == self.current_objective_color:
                        return move
                    if move.action == 1 and move.colorToDraw == 'WILD':
                        draw_wild_move = move
                    if move.action == 2:
                        draw_top_move = move
            else:
                max_color = {x: game.players[game.turn%len(game.players)].trainCardHand.count(x) for x in color_indexing.keys() if x != 'WILD'}
                max_color = max(max_color.items(), key=operator.itemgetter(1))

                for move in draw_train_card_moves:
                    if move.action == 1 and move.colorToDraw == max_color:
                        return move
                    if move.action == 1 and move.colorToDraw == 'WILD':
                        draw_wild_move = move
                    if move.action == 2:
                        draw_top_move = move
            
            if draw_wild_move != None:
                return draw_wild_move
            if draw_top_move != None:
                return draw_top_move
            
        if len(draw_train_card_moves) > 0:
            return random.choice(draw_train_card_moves)
        if len(claim_route_moves) > 0:
            return random.choice(claim_route_moves)
        
        return random.choice(possible_moves)
    
    def generate_game_plan(self, game: Game):
        joint_graph = self.joint_graph(game, 3 if game.players[game.turn % len(game.players)].trainsLeft >= 3 else game.players[game.turn % len(game.players)].trainsLeft)
        city1 = None
        city2 = None
        city3 = None
        color = None
        min_trains_threshold = 8

        list_of_destinations = self.destinations_not_completed(game, joint_graph)
        if list_of_destinations:
            most_valuable_route_index = max(range(len(list_of_destinations)), key=lambda index: list_of_destinations[index]['points'])
            most_valuable_route = list_of_destinations[most_valuable_route_index]
            result = self.chooseNextRouteTarget(game, joint_graph, most_valuable_route['city1'], most_valuable_route['city2'])
            if result != False:
                city1, city2, color = result
        if city1 == None:
            result = self.chooseMaxRoute(game)
        
        return result

    def chooseMaxRoute(self, game:Game):
        number_of_trains_left = game.players[game.turn % len(game.players)].trainsLeft
        max_size = 0
        list_of_edges = []

        free_routes_graph = self.free_routes_graph(game.board, len(game.players))
        for city1 in free_routes_graph:
            for city2 in free_routes_graph[city1]:
                for e in free_routes_graph[city1][city2]:
                    edge = free_routes_graph[city1][city2][e]
                    if edge['weight'] <= number_of_trains_left:
                        if edge['weight'] > max_size:
                            max_size = edge['weight']
                            list_of_edges = [(edge, city1, city2)]
                        elif edge['weight'] == max_size:
                            list_of_edges.append((edge, city1, city2))
        
        if len(list_of_edges) > 0:
            best_route = [self.rank(x[0], game) for x in list_of_edges]
            best_route = list_of_edges[best_route.index(max(best_route))]

            return [best_route[1], best_route[2], best_route[0]['color']]

        return [None, None, None]

    def rank(self, edge, game: Game):
        color = edge['color']
        player_colors_no_wild = {x: game.players[game.turn%len(game.players)].trainCardHand.count(x) for x in color_indexing.keys() if x != 'WILD'}
        number_of_wilds = game.players[game.turn%len(game.players)].trainCardHand.count("WILD")
        max_color_value = max(player_colors_no_wild.values())

        if color == 'GRAY':
            if max_color_value >= edge['weight']:
                return 15
            if max_color_value + number_of_wilds >= edge['weight']:
                return 9
            return 10 - edge['weight'] + max_color_value
        
        if player_colors_no_wild[color] >= edge['weight']:
            return 10
        if player_colors_no_wild[color] + number_of_wilds >= edge['weight']:
            return 9
        
        return 9 - edge['weight'] + max_color_value

    def chooseNextRouteTarget(self, game: Game, graph: nx.MultiGraph, city1, city2):
        try:
            list_of_route_nodes = nx.shortest_path(graph, city1, city2)
        except:
            return False
        
        list_of_colors = set()
        cities = []
        for i in range(0, len(list_of_route_nodes)-1):
            cities = [list_of_route_nodes[i], list_of_route_nodes[i+1]]
            for key in graph[list_of_route_nodes[i]][list_of_route_nodes[i+1]]:
                edge = graph[list_of_route_nodes[i]][list_of_route_nodes[i+1]][key]

                if edge['owner'] != None:
                    list_of_colors = set()
                    cities = []
                    break

                list_of_colors.add(edge['color']) # BUG: .lower() was left out
        
            if len(cities) != 0:
                break
        
        color_weight = []
        list_of_colors = list(list_of_colors)
        if 'GRAY' in list_of_colors:
            list_of_colors = [x for x in color_indexing.keys()]
        for color in list_of_colors:
            color_weight.append(game.players[game.turn%len(game.players)].trainCardHand.count(color))
        
        max_weight = color_weight.index(max(color_weight))
        desired_color = list_of_colors[max_weight]

        return [cities[0], cities[1], desired_color]

    def destinations_not_completed(self, game: Game, joint_graph: nx.MultiGraph):
        result = []

        graph = nx.Graph()
        for node1 in game.board:
            for node2 in game.board[node1]:
                for edge in game.board[node1][node2]:
                    if game.board[node1][node2][edge]['owner'] == game.turn % len(game.players):
                        graph.add_edge(node1, node2, weight=game.board[node1][node2][edge]['weight'])
        
        destination_cards = game.players[game.turn % len(game.players)].destinationCardHand
        for card in destination_cards:
            city1 = card.city1
            city2 = card.city2
            try:
                nx.shortest_path(graph, city1, city2)
                solved = True
            except:
                solved = False
            
            if not solved:
                if city1 in joint_graph.nodes() and city2 in joint_graph.nodes() and nx.has_path(joint_graph, city1, city2):
                    result.append({'city1': city1, 'city2': city2, 'points': card.points})
        
        return result

    def chooseDestinations(self, moves: list[Action], game: Game, min_weight_edge: int):
        best_move = (0, None)
        least_worst_move = (0, None)
        joint_graph = self.joint_graph(game, min_weight_edge)

        for m in moves:
            current_move_value = 0
            number_of_trains_needed = 0
            points = 0
            for ind in m.takeDests:
                temp = self.calculate_value([game.destinationCardsDealt[ind].city1, game.destinationCardsDealt[ind].city2], game.destinationCardsDealt[ind].points, joint_graph)
                current_move_value += temp[0]
                number_of_trains_needed += temp[1]
                points += game.destinationCardsDealt[ind].points
            
            if number_of_trains_needed <= game.players[game.turn % len(game.players)].trainsLeft:
                total = current_move_value / number_of_trains_needed
                if total > best_move[0]:
                    best_move = (total, m)
            else:
                if least_worst_move[1] == None:
                    least_worst_move = (points, m)
                else:
                    if least_worst_move[0] > points:
                        least_worst_move = (points, m)
        
        if best_move[1] != None:
            return best_move[1]

        return least_worst_move[1]

    def calculate_value(self, cities: list[str], points: int, graph: nx.MultiGraph):
        try:
            if nx.has_path(graph, cities[0], cities[1]):
                left_to_claim = 0
                path = nx.shortest_path(graph, cities[0], cities[1])
                for s in range(0, len(path)-1):
                    for t in range(s+1, len(path)):
                        temp = 0
                        for edge in graph[path[s]][path[t]]:
                            if edge['owner'] == None:
                                temp = edge['weight']
                            else:
                                temp = 0
                                break
                        
                        left_to_claim = left_to_claim + temp
                return [float(points), float(left_to_claim)]
            else:
                return [float(-1.0 * points), float(50)]
        except:
            return [float(-1.0 * points), float(50)]

    def joint_graph(self, game: Game, min_weight_edge=0):
        free_connections_graph = self.free_routes_graph(game.board, len(game.players), min_weight_edge)
        player_edges = [edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.turn % len(game.players)]
    
        joint_graph = free_connections_graph
        for edge in player_edges:
            joint_graph.add_edge(edge[0], edge[1], weight=0, color='none', owner=game.turn % len(game.players))
        
        return joint_graph

    def free_routes_graph(self, graph: nx.MultiGraph, number_of_players: int, min_weight_edge: int=0):
        G = nx.MultiGraph()

        visited_nodes = []
        # BUG: make sure that this is generating not just each free route... but the takeable ones...
        # even if a route is free, if the player in question has the other route between the two cities taken
        # then it is not free...
        for node1 in graph:
            for node2 in graph[node1]:
                if node2 not in visited_nodes:
                    locked = False
                    if not locked:
                        for edge in graph[node1][node2]:
                            if graph[node1][node2][edge]['owner'] == None and graph[node1][node2][edge]['weight'] >= min_weight_edge:
                                G.add_edge(node1, node2, weight=graph[node1][node2][edge]['weight'], color=graph[node1][node2][edge]['color'], owner=None)
            visited_nodes.append(node1)
        
        return G