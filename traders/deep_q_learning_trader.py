import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from framework.portfolio import Portfolio
from framework.stock_data import StockData
from framework.vote import Vote
import numpy as np
import enum
from enum import Enum


class ChosenAction(Enum):
    RAND = "random"
    MAX_Q = "largestQ"


class State:
    portfolio: Portfolio
    expert_vote_stock_a: Vote
    expert_vote_stock_b: Vote
    current_stock_data_a: StockData
    current_stock_data_b: StockData

    # states anzahl wert portofilio aktien experten meinungen
    def __init__(self, portfolio: Portfolio, expert_vote_stock_a: Vote, expert_vote_stock_b: Vote, current_stock_data_a: StockData, current_stock_data_b: StockData):
        self.portfolio = portfolio
        self.expert_vote_stock_a = expert_vote_stock_a
        self.current_stock_data_a = current_stock_data_a
        self.expert_vote_stock_b = expert_vote_stock_b
        self.current_stock_data_b = current_stock_data_b

    def get_nn_input_state(self):
        vote_a = 0
        vote_b = 0
        if self.expert_vote_stock_a == Vote.BUY:
            vote_a = 1
        elif self.expert_vote_stock_a == Vote.HOLD:
            vote_a = 2
        else:
            vote_a = 3
        if self.expert_vote_stock_b == Vote.BUY:
            vote_b = 1
        elif self.expert_vote_stock_b == Vote.HOLD:
            vote_b = 2
        else:
            vote_b = 3
        return np.array([[vote_a, vote_b]])


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading
        self.actions = [
            (+0.0, +1.0),
            (+0.0, -1.0),
            (+0.0, +0.0),
            (+1.0, +0.0),
            (-1.0, +0.0),
            (-1.0, -1.0),
            (-1.0, +1.0),
            (+1.0, -1.0),
            (+0.25, +0.75),
            (+0.50, +0.50),
            (+0.75, +0.25),
            (-0.25, -0.75),
            (-0.50, -0.50),
            (-0.75, -0.25),
            (-0.50, +0.50),
            (+0.50, -0.50),
            (-0.25, +0.25),
            (+0.25, -0.25)
        ]
        # Parameters for neural network
        self.state_size = 2
        self.action_size = len(self.actions)
        self.hidden_size = 50
        self.discount = 0.9
        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None
        self.reward_factor = 100
        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def gen_reward(self, portfolio: Portfolio, stock_market_data: StockMarketData):
        print('gen_reward')
        if self.last_portfolio_value < portfolio.get_value(stock_market_data):
            return self.reward_factor*(portfolio.get_value(stock_market_data)/self.last_portfolio_value)
        elif self.last_portfolio_value > portfolio.get_value(stock_market_data):
            return -self.reward_factor*(portfolio.get_value(stock_market_data)/self.last_portfolio_value)
        else:
            return -self.reward_factor/5

    def update_memory(self, reward, current_state_param):
        print('update_memory')
        current_memory_state = (self.last_state, self.last_action, reward, current_state_param)
        self.memory.append(current_memory_state)

    def train_model(self):
        if len(self.memory) > self.min_size_of_memory_before_training + self.batch_size:
            rand_memories = random.sample(self.memory, self.batch_size)
            for last_state, last_action, reward, current_state_param in rand_memories:
                curr_q_vec = reward + self.discount * np.amax(self.model.predict(current_state_param)[0])
                q_vec = self.model.predict(last_state)
                q_vec[0][last_action] = curr_q_vec
                self.model.fit(last_state, q_vec, epochs=2, verbose=1)
        print('model trained')

    def calc_new_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def get_action_idx(self, curr_state):
        chosen_behavior = np.random.choice(
            a=[ChosenAction.RAND, ChosenAction.MAX_Q], size=1, p=[self.epsilon, 1-self.epsilon]
        )[0]
        idx = self.gen_action_index(chosen_behavior, curr_state)

        self.calc_new_epsilon()
        return idx

    def gen_action_index(self, chosen_behavior, curr_state):
        if not self.train_while_trading or chosen_behavior == ChosenAction.MAX_Q:
            prediction = self.model.predict(curr_state)
            return np.argmax(prediction[0])
        return np.random.randint(self.action_size)

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]
        state = State(portfolio, self.expert_a.vote(stock_market_data[Company.A]), self.expert_b.vote(stock_market_data[Company.B]), stock_market_data[Company.A], stock_market_data[Company.B])
        param = state.get_nn_input_state()
        if self.train_while_trading and self.last_state is not None:
            current_reward = self.gen_reward(portfolio, stock_market_data)
            self.update_memory(current_reward, param)
            self.train_model()

        self.last_action = self.get_action_idx(param)
        self.last_state = param
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        orders = self.get_orders(stock_market_data, portfolio)
        return orders

    def get_orders(self, stock_market_data: StockMarketData, portfolio: Portfolio):
        orders = []
        price_a = stock_market_data[Company.A].get_last()[-1]
        sell_off_a = portfolio.get_stock(Company.A)
        action_a = self.actions[self.last_action][0]
        order_a = int(action_a * portfolio.cash // price_a)
        orders.append(self.get_order_item(action_a, order_a, sell_off_a, Company.A))

        price_b = stock_market_data[Company.B].get_last()[-1]
        sell_off_b = portfolio.get_stock(Company.B)
        action_b = self.actions[self.last_action][1]
        order_b = int(action_b * portfolio.cash // price_b)
        orders.append(self.get_order_item(action_b, order_b, sell_off_b, Company.B))
        return orders

    def get_order_item(self, action, order, sell_off, company):
        if action > 0:
            return Order(OrderType.BUY, company, order)
        else:
            return Order(OrderType.SELL, company, sell_off)


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 2
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
