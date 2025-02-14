import random
import operator
import networkx as nx
from engine.game import Game
from engine.objects import Action

class HungryAgent:
    def __init__(self):
        self.players_previous_points = 0
        self.colors_needed = {}
        self.routes_by_color = {}
        self.current_threshold = 0
    
    def decide(self, game: Game):
        possible_moves = game.getValidMoves()
        if len(possible_moves) == 0:
            print("No Possible Moves")
            return None
        
        free_connections_graph = self.free_routes_graph(game.board, len(game.players))
        player_edges = [edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.turn % len(game.players)]
        
        joint_graph = free_connections_graph
        for edge in player_edges:
            joint_graph.add_edge(edge[0], edge[1], weight=0, color='none')
        
        list_of_cities: list[str] = []
        d_points = 0
        for d in game.players[game.turn % len(game.players)].destinationCardHand:
            if d.city1 in joint_graph and d.city2 in joint_graph:
                list_of_cities.extend([d.city1, d.city2])
                d_points += d.points

        if possible_moves[0].action == 3 and possible_moves[0].askingForDeal == False:
            best_ratio = 0
            min_over_train_requirements = game.players[game.turn % len(game.players)].trainsLeft
            best_move = None
            min_points = None
            for m in possible_moves:
                destinations = list(list_of_cities)
                temp = []
                points = d_points
                for d in game.destinationCardsDealt:
                    points += d.points
                    destinations.extend([d.city1, d.city2])
                if min_points == None or points < min_points:
                    min_points = points
                    min_move = m
                destinations = list(set(destinations))
                x = self.generate_game_plan(destinations, joint_graph)
                fitness = 0
                if x[0] != None:
                    fitness = float((points + x[0])) / float(x[1])
                if x[1] <= game.players[game.turn % len(game.players)].trainsLeft - 5:
                    if fitness > best_ratio:
                        best_ratio = fitness
                        best_move = m
                        croutes = x[2]
                        cneeded = x[3]
                elif x[1] <= game.players[game.turn % len(game.players)].trainsLeft:
                    if x[1] < min_over_train_requirements:
                        min_over_train_requirements = x[1]
                        best_move = m
                        croutes = x[2]
                        cneeded = x[3]
            
            if best_move == None:
                return min_move

            self.colors_needed = cneeded
            self.routes_by_color = croutes
            self.current_threshold = x[1]
            for i in range(0, len(game.players)):
                if i != (game.turn % len(game.players)):
                    self.players_previous_points += game.players[i].points
            return best_move

        total_current_points = 0
        for i in range(0, len(game.players)):
            total_current_points += game.players[i].points

        if self.players_previous_points < total_current_points:
            x = self.generate_game_plan(list_of_cities, joint_graph)
            self.colors_needed = x[3]
            self.routes_by_color = x[2]
            self.players_previous_points = total_current_points
        
        if self.current_threshold < game.players[game.turn % len(game.players)].trainsLeft - 8:
            for move in possible_moves:
                if move.action == 3 and move.askingForDeal == True:
                    return move
        
        routes_to_take: list[Action] = []
        for move in possible_moves:
            if move.action == 0:
                routes_to_take.append(move)
        
        max_route_move = None
        max_route_value = None

        total = 0
        if self.routes_by_color != None:
            total = sum(len(x) for x in self.routes_by_color.values())
        if total <= 0 and len(routes_to_take) > 0:
            for action in routes_to_take:
                if action.route.city1 in free_connections_graph:
                    if action.route.city2 in free_connections_graph[action.route.city1]:
                        for key in free_connections_graph[action.route.city1][action.route.city2]:
                            edge = free_connections_graph[action.route.city1][action.route.city2][key]
                            if edge['color'] == action.route.color or edge['color'] == 'GRAY':
                                if max_route_value == None or max_route_value < edge['weight']:
                                    max_route_value = edge['weight']
                                    max_route_move = move
            return max_route_move
        
        if len(routes_to_take) > 0:
            for action in routes_to_take:
                if action.route.city1 < action.route.city2:
                    temp1 = action.route.city1
                    temp2 = action.route.city2
                else:
                    temp1 = action.route.city2
                    temp2 = action.route.city1
                
                if [temp1, temp2] in self.routes_by_color[action.route.color] or [temp2, temp1] in self.routes_by_color[action.route.color]:
                    return action
                
                if [temp1, temp2] in self.routes_by_color['GRAY'] or [temp2, temp1] in self.routes_by_color['GRAY']:
                    if self.colors_needed[action.route.color] <= 0:
                        return action

        moves_by_color = {}
        for move in possible_moves:
            if move.action == 1:
                moves_by_color[move.colorToDraw.upper()] = move
            elif move.action == 2:
                moves_by_color["TOP"] = move
        
        colors_available = game.faceUpCards.copy()
        max_color_available = max(colors_available.items(), key=operator.itemgetter(1))
        if self.colors_needed != None:
            most_needed_color = max(self.colors_needed.items(), key=operator.itemgetter(1))[0]
        else:
            most_needed_color = 'NONE'
        
        if most_needed_color in colors_available:
            return moves_by_color[most_needed_color.upper()]
        
        if self.colors_needed != None and self.colors_needed[max_color_available[0].upper()] > 0 and max_color_available[0].upper() in moves_by_color:
            return moves_by_color[max_color_available[0].upper()]
        
        if most_needed_color == 'GRAY' and max_color_available[1] > 1:
            if max_color_available[0].upper() in moves_by_color:
                return moves_by_color[max_color_available[0].upper()]
        
        if 'TOP' in moves_by_color:
            return moves_by_color['TOP']

        return random.choice(possible_moves)

    def generate_game_plan(self, dkey_nodes: list[str], G: nx.MultiGraph):
        longest_route = None
        size_longest_route = 0

        result = {'start': set(), 'end': set()}

        for x in range(0, len(dkey_nodes)-1):
            for y in range(x+1, len(dkey_nodes)):
                try:
                    if nx.has_path(G, dkey_nodes[x], dkey_nodes[y]):
                        temp_route_size = nx.dijkstra_path_length(G, dkey_nodes[x], dkey_nodes[y])
                        if temp_route_size > size_longest_route:
                            size_longest_route = temp_route_size
                            result['start'] = set([dkey_nodes[x]])
                            result['end'] = set([dkey_nodes[y]])
                except:
                    pass
        
        key_nodes = list((set(dkey_nodes) - result['start']) - result['end'])

        where = ''
        size_shortest_route = None
        which = []
        routes_dict = {}
        total_points_from_routes = 0
        for x in key_nodes:
            for y in result['start']:
                try:
                    temp_route_size = nx.dijkstra_path_length(G, x, y)
                    if size_shortest_route == None or temp_route_size < size_shortest_route:
                        size_shortest_route = temp_route_size
                        which = [x, y]
                        where = 'start'
                except:
                    pass
            
            for y in result['end']:
                try:
                    temp_route_size = nx.dijkstra_path_length(G, x, y)
                    if size_shortest_route == None or temp_route_size < size_shortest_route:
                        size_shortest_route = temp_route_size
                        which = [x, y]
                        where = 'end'
                except:
                    pass
            
            if where == '':
                return [None, None, None, None]
            result[where] = result[where] | set([x])
            try:
                temp_path = nx.dijkstra_path(G, x, which[1])
            except:
                temp_path = []
            
            for i in range(0, len(temp_path)-1):
                if temp_path[i] > temp_path[i+1]:
                    temp1 = temp_path[i+1]
                    temp2 = temp_path[i]
                else:
                    temp1 = temp_path[i]
                    temp2 = temp_path[i+1]
                
                if (temp1 not in routes_dict) and (temp2 not in routes_dict):
                    routes_dict[temp1] = [temp2]
                
                elif (temp1 in routes_dict):
                    if temp2 not in routes_dict[temp1]:
                        routes_dict[temp1].append(temp2)
                else:
                    if temp1 not in routes_dict[temp2]:
                        routes_dict[temp2].append(temp1)
        
        size_shortest_route = None
        for x in result['start']:
            for y in result['end']:
                try:
                    temp_route_size = nx.dijkstra_path_length(G, x, y)
                    if size_shortest_route == None or temp_route_size < size_shortest_route:
                        size_shortest_route = temp_route_size
                        which = [x, y]
                except:
                    temp_route_size = 0
        try:
            temp_path = nx.dijkstra_path(G, which[0], which[1])
        except:
            temp_path = []
        
        for i in range(0, len(temp_path)-1):
            if temp_path[i] > temp_path[i+1]:
                temp1 = temp_path[i+1]
                temp2 = temp_path[i]
            else:
                temp1 = temp_path[i]
                temp2 = temp_path[i+1]
            
            if (temp1 not in routes_dict) and (temp2 not in routes_dict):
                routes_dict[temp1] = [temp2]
            
            elif (temp1 in routes_dict):
                if temp2 not in routes_dict[temp1]:
                    routes_dict[temp1].append(temp2)
            else:
                if temp1 not in routes_dict[temp2]:
                    routes_dict[temp2].append(temp1)

        colors_needed = {"BLUE": 0, "GREEN": 0, "RED": 0, "PINK": 0, "ORANGE": 0, "BLACK": 0, "YELLOW": 0, "WHITE": 0, "GRAY": 0, "WILD": 0}
        color_routes = {"BLUE": [], "GREEN": [], "RED": [], "PINK": [], "ORANGE": [], "BLACK": [], "YELLOW": [], "WHITE": [], "GRAY": []}
        double_opt = []
        point_dict = {1:1, 2:2, 3:4, 4:7, 5:10, 6:15, 8:21, 9:27}

        for key in routes_dict:
            for x in routes_dict[key]:
                if len(G[key][x].keys()) > 1:
                    temp = []
                    owned = False
                    for y in G[key][x]:
                        edge = G[key][x][y]
                        if edge['weight'] == 0:
                            owned = True
                            break
                        temp.append((edge['color'], edge['weight'], key, x))
                    if not owned:
                        double_opt.append(temp)

                else:
                    edge = G[key][x][0]
                    if edge['weight'] > 0:
                        colors_needed[edge['color']] += edge['weight']
                        # colors_needed['WILD'] += edge['ferries']
                        color_routes[edge['color']].append([key, x])
                        total_points_from_routes += point_dict[edge['weight']]
        
        for edge_list in double_opt:
            min_val = 0
            max_color = None
            temp = None
            flag = False
            for (color, weight, city1, city2) in edge_list:
                if colors_needed[color] == 0:
                    colors_needed[color] += weight
                    # colors_needed['WILD'] += ferries
                    color_routes[color].append([city1, city2])
                    total_points_from_routes += point_dict[weight]
                    flag = True
                    break
                else:
                    if max_color == None or colors_needed[color] < min_val:
                        max_color = color
                        min_val = colors_needed[color]
                        temp = (color, weight, city1, city2)

            if not flag:
                colors_needed[temp[0]] += temp[1]
                # colors_needed['WILD'] += temp[2]
                color_routes[temp[0]].append([temp[2], temp[3]])
                total_points_from_routes += point_dict[weight]

        return [total_points_from_routes, sum(colors_needed.values()), color_routes, colors_needed]

    def free_routes_graph(self, graph: nx.MultiGraph, number_of_players: int):
        G = nx.MultiGraph()

        visited_nodes = []

        for node1 in graph:
            for node2 in graph[node1]:
                if node2 not in visited_nodes:
                    locked = False
                    if not locked:
                        for edge in graph[node1][node2]:
                            if graph[node1][node2][edge]['owner'] == None:
                                G.add_edge(node1, node2, weight=graph[node1][node2][edge]['weight'], color=graph[node1][node2][edge]['color'])

            visited_nodes.append(node1)
        
        return G