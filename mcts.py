import math
import random
import numpy as np
import simulator
import status
from sys import float_info
from copy import deepcopy
from board import Board
from status import GameStatus, Chess


class Node(object):
    def __init__(self, status: Board, action=None, parent=None):
        self.__status = status
        self.__action = action
        self.__total_game = 0
        self.__won_game = 0
        self.__parent = parent
        self.__children = []

    @staticmethod
    def __find_best_node(node_list):
        best_rate, best_node = -1, None
        for i in node_list:
            if i.__total_game and i.__won_game / i.__total_game > best_rate:
                best_rate = i.__won_game / i.__total_game
                best_node = i
        return best_node

    @property
    def action(self):
        return self.__action

    @property
    def win_prob(self):
        return self.__won_game / self.__total_game \
            if self.__total_game else 0
            
   
    def search(self, depth=20, breadth=3):
        target = status.get_won_status(self.__status.status)
        for _ in range(depth):
            node = self
            while node.__children:
                node = node.select_stochastic()
            if node is self or node.__total_game:
                child = node.expand(breadth)
            else:
                child = node
            if child:
                value = child.playout(target)
                child.update(value)
            else:
                node.update(None)
        return self.__find_best_node(self.__children).__action

    def selectSeed(self, seed=100):
        random.seed(seed)
        selected = None
        best_value = -1.0
        for i in self.__children:
            uct = i.__won_game / (i.__total_game + float_info.epsilon)
            uct += math.sqrt(math.log(self.__total_game + 1)
                            / (i.__total_game + float_info.epsilon))
            exploration_factor = math.sqrt(math.log(self.__total_game + 1) / (i.__total_game + float_info.epsilon))
            uct += random.random() * exploration_factor
            if uct > best_value:
                selected = i
                best_value = uct
        return selected
    
    def select_stochastic(self, seed=0):
        total_visits = sum(child.__total_game for child in self.__children)
        np.random.seed(1)
        exploration_factor = math.sqrt(math.log(total_visits) / sum(child.__total_game for child in self.__children))
        ucb_values = []
        for child in self.__children:
            if child.__total_game == 0:
                ucb_values.append(1)
            else:
                ucb_values.append(child.__won_game / child.__total_game + exploration_factor)
        if np.array(ucb_values) / sum(ucb_values) is np.nan:
            return np.random.choice(self.__children)
        return np.random.choice(self.__children, p=np.array(ucb_values) / sum(ucb_values))

    def print_tree(self, tab=0):
        tab_str = '| ' * (tab - 1) + ('+-' if tab else '')
        print(tab_str + 'won/total: %d/%d' %
              (self.__won_game, self.__total_game),
              end=', ')
        print('action:', str(self.__action), end=', ')
        print('status:', str(self.__status.status))
        for i in self.__children:
            i.print_tree(tab + 1)

    def select(self):
        selected = None
        best_value = -1.0
        for i in self.__children:
            uct = i.__won_game / (i.__total_game + float_info.epsilon)
            uct += math.sqrt(math.log(self.__total_game + 1)
                             / (i.__total_game + float_info.epsilon))
            uct += random.random() * float_info.epsilon
            if uct > best_value:
                selected = i
                best_value = uct
        return selected
    
    def selectEpsilon(self, epsilon=0.1):
        if random.random() < epsilon:
            # Con probabilidad ε, seleccionamos una acción al azar.
            selected = random.choice(self.__children)
        else:
            # Con probabilidad 1-ε, seleccionamos la acción que actualmente parece ser la mejor.
            selected = None
            best_value = -1.0
            for i in self.__children:
                uct = i.__won_game / (i.__total_game + float_info.epsilon)
                uct += math.sqrt(math.log(self.__total_game + 1)
                                / (i.__total_game + float_info.epsilon))
                if uct > best_value:
                    selected = i
                    best_value = uct
        return selected
        

    def selectTemperature(self, temperature=1.0):
        uct_values = []
        for i in self.__children:
            uct = i.__won_game / (i.__total_game + 1.0)
            uct += math.sqrt(math.log(self.__total_game + 1)
                            / (i.__total_game + 1.0))
            uct_values.append(uct)

        # Convertimos los valores UCT a probabilidades usando una función softmax
        uct_values = np.array(uct_values)
        uct_values -= np.max(uct_values)
        probs = np.exp(uct_values / temperature)
        probs /= np.sum(probs)
        
        # Seleccionamos un hijo basado en las probabilidades
        selected = np.random.choice(self.__children, p=probs)

        return selected

    def expand(self, breadth=3):
        if self.__status.won:
            return None
        else:
            actions = []
            action = simulator.random_action(self.__status)
            for _ in range(breadth):
                try_count = 0
                while action in actions:
                    action = simulator.random_action(self.__status)
                    try_count += 1
                    if try_count > breadth * 2:
                        return random.choice(self.__children)
                next_status = deepcopy(self.__status)
                next_status.apply_action(action)
                actions.append(action)
                self.__children.append(Node(next_status, action, self))
            return random.choice(self.__children)

    def playout(self, target_status):
        return simulator.simulate(self.__status, target_status)

    def update(self, value):
        if value is None:
            value = 1
        node = self
        while node:
            node.__total_game += 1
            node.__won_game += value
            node = node.__parent


# some test
if __name__ == '__main__':
    board = Board()
    while True:
        node = Node(board)
        depth = 20 if board.status == GameStatus.RedMoving else 10
        board.apply_action(node.search(depth=depth, breadth=int(depth/4)))
        print(board)
        # node.print_tree()
        try:
            input()
        except KeyboardInterrupt:
            break
        except EOFError:
            break