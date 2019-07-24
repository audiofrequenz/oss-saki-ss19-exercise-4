from framework.portfolio import Portfolio
from framework.stock_data import StockData
from framework.vote import Vote
import numpy as np

class State:
    portfolio: Portfolio
    expert_vote_stock_a: Vote
    expert_vote_stock_b: Vote
    current_stock_data_a: StockData
    current_stock_data_b: StockData
    #previous_stock_data_a: StockData
    #previous_stock_data_b: StockData

    # states anzahl wert portofilio aktien experten meinungen
    def __init__(self, portfolio: Portfolio, expert_vote_stock_a: Vote, expert_vote_stock_b: Vote, current_stock_data_a: StockData, current_stock_data_b: StockData):
        self.portfolio = portfolio
        self.expert_vote_stock_a = expert_vote_stock_a
        self.current_stock_data_a = current_stock_data_a
        #self.previous_stock_data_a = previous_stock_data_a
        self.expert_vote_stock_b = expert_vote_stock_b
        self.current_stock_data_b = current_stock_data_b
        #self.previous_stock_data_b = previous_stock_data_b

    def get_nn_input_state(self):
        voteA = 0
        voteB = 0
        if self.expert_vote_stock_a == Vote.BUY:
            voteA = 1
        elif self.expert_vote_stock_a == Vote.HOLD:
            voteA = 2
        else:
            voteA = 3
        if self.expert_vote_stock_b == Vote.BUY:
            voteB = 1
        elif self.expert_vote_stock_b == Vote.HOLD:
            voteB = 2
        else:
            voteB = 3
        return np.array([[voteA, voteB]])
        #return np.array([[expert vote a, expert vote b]])
    def get_nn_input_param(self):
        param = np.zeros(2)
        param = np.zeros(9)
        if self.expert_vote_stock_a == Vote.BUY:
            param[0] = 1
        elif self.expert_vote_stock_a == Vote.HOLD:
            param[1] = 1
        else:
            param[2] = 1

        param[3] = (1/self.current_stock_data_a.get_from_offset(self.current_stock_data_a.get_row_count()-2)[0][1])*self.current_stock_data_a.get_last()[1]
        if self.portfolio.cash > 0:
            param[4] = 1
        else:
            param[4] = 0
        if self.expert_vote_stock_b == Vote.BUY:
            param[5] = 1
        elif self.expert_vote_stock_b == Vote.HOLD:
            param[6] = 1
        else:
            param[7] = 1

        param[8] = (1/self.current_stock_data_a.get_from_offset(self.current_stock_data_a.get_row_count()-2)[0][1])*self.current_stock_data_a.get_last()[1]
        return param
