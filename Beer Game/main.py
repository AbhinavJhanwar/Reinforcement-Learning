import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import _pickle as pickle
import configparser
np.random.seed(42)


# environment class
class SupplyChain:
    def __init__(self, inventory, backlog, incoming_delivery):
        # initializing various parameters of supply chain
        self.inventory = [inventory, -1]
        self.backlog = [backlog, -1]
        self.incoming_delivery = incoming_delivery
        self.outgoing_delivery = -1
        self.incoming_order = -1
        self.outgoing_order = []
        for i in range(time_delay):
            self.outgoing_order.append(10)
        self.week = 0
        self.upstream_pending_delivery = 0
        self.total_cost = 0

    def set_cost(self, inventory_cost, backlog_cost):
        # set cost parameters of supply chain
        self.inventory_cost = inventory_cost
        self.backlog_cost = backlog_cost

    def get_reward(self):
        # reward of a particular state
        return (-max(self.inventory[1]*self.inventory_cost, min_cost) +
                self.backlog[1]*self.backlog_cost)

    def current_state(self):
        # get the current state of environment. Set the current inventory
        # and backlog value as per their actual value to a category
        # between -n_bins and +n_binsit also makes sure that min state
        # value is -n_bins and max state value is +n_bins
        x = min(n_bins, max(-n_bins, int((self.inventory[1] -
                self.backlog[1])*n_bins/max_inventory)))
        return x

    def update(self):
        # get outgoing_delivery for current state, inventory &
        # backlog of next state
        if self.incoming_order >= self.inventory[0] + self.incoming_delivery:
            self.outgoing_delivery = self.inventory[0] + self.incoming_delivery
            self.backlog[1] = (self.backlog[0] + self.incoming_order -
                               self.outgoing_delivery)
            self.inventory[1] = 0
        elif self.incoming_order <= (self.inventory[0] +
                                     self.incoming_delivery - self.backlog[0]):
            self.outgoing_delivery = self.incoming_order + self.backlog[0]
            self.backlog[1] = 0
            self.inventory[1] = (self.inventory[0] + self.incoming_delivery -
                                 self.outgoing_delivery)
        elif self.incoming_order > (self.inventory[0] +
                                    self.incoming_delivery - self.backlog[0]):
            self.outgoing_delivery = self.inventory[0] + self.incoming_delivery
            self.backlog[1] = (self.backlog[0] + self.incoming_order -
                               self.outgoing_delivery)
            self.inventory[1] = 0

    def clock_tick(self):
        # modify the state of the environment as per the action taken by
        # the agent

        # get cost of current state for the given action(outgoing_order)
        cost = self.get_reward()

        # get incoming delivery for next cycle
        if random_incoming_delivery:
            # get random incoming delivery
            self.incoming_delivery, self.upstream_pending_delivery = \
                random_delivery(
                    self.outgoing_order[0], self.upstream_pending_delivery)
        else:
            # set incoming delivery
            self.incoming_delivery = self.outgoing_order[0]

        # update next week values to current week
        self.backlog[0] = self.backlog[1]
        self.inventory[0] = self.inventory[1]
        # update outgoing order based on time delay
        for i in range(time_delay-1):
            self.outgoing_order[i] = self.outgoing_order[i+1]

        # increment week to take the current state to next state or time period
        self.week += 1

        # return the cost of current state of environment
        return cost

    def year_over(self):
        # check based on the number of weeks if year end is reached
        return self.week == number_of_weeks

    def all_states(self):
        # get all the possible states of the environment
        return list(range(-n_bins, n_bins+1))

    def update_data_log(self, warehouse_log):
        # update the dataframe for various parameters of supply chain

        # set total cost by adding all costs for the episode
        self.total_cost += self.get_reward()

        # append values of current state parameters in dataframe
        warehouse_log = warehouse_log.append({
                        'Current_Inventory': self.inventory[0],
                        'Backlog_Orders': self.backlog[0],
                        'Incoming_Delivery': self.incoming_delivery,
                        'Outgoing_Delivery': self.outgoing_delivery,
                        'Outgoing_Order': self.outgoing_order[time_delay-1],
                        'Incoming_Order': self.incoming_order,
                        'Closing_Inventory': self.inventory[1],
                        'Closing_Backlog': self.backlog[1],
                        'State': self.current_state(),
                        'Pending_Delivery': self.upstream_pending_delivery,
                        'Total_Cost': abs(self.total_cost),
                        'week': self.week
                        }, ignore_index=True)

        return warehouse_log


def Warehouse(
        inventory=400, backlog=0, incoming_delivery=10,
        inventory_cost=0.5, backlog_cost=1):
    # create and return the environment for the reinforcement learning

    # initialize states
    env = SupplyChain(
        inventory=inventory, backlog=backlog,
        incoming_delivery=incoming_delivery)

    # set holding, backlog cost
    env.set_cost(inventory_cost=inventory_cost, backlog_cost=backlog_cost)

    return env


def random_action(a, eps=0.1):
    # return a random action based on epsilon greedy algorithm
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(all_possible_actions)


def random_delivery(delivery, pending):
    # return number of deliveries from upstream and pending order
    delivery_new = np.random.randint(low=pending, high=delivery+pending+1)
    pending_new = delivery + pending - delivery_new
    return delivery_new, pending_new


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    max_value = np.max(d)
    max_key = np.argmax(d)
    return max_key, max_value


def get_incoming_order(mean, sd):
    # function to get the customer order
    return int(np.random.normal(mean, sd))


def get_incoming_order_stats(df, column):
    df = pd.read_csv(df)
    return df[[column]].mean(), df[[column]].std()


def Q_learning(s, a, Q, warehouse, eps, warehouse_log, mean, sd, sa_count):
    # get outgoing order/action for the current state
    action = all_possible_actions[a]
    action = random_action(action, eps)   # epsilon greedy

    # set outgoing_order based on the action to be taken
    warehouse.outgoing_order[time_delay-1] = action

    # update the warehouse dataframe
    warehouse_log = warehouse.update_data_log(warehouse_log)

    # get cost of current week & update the next week values as current week
    r = warehouse.clock_tick()

    # get incoming order
    warehouse.incoming_order = get_incoming_order(mean, sd)

    # update outgoing delivery and next inventory, backlog
    warehouse.update()

    # get next state
    s1 = warehouse.current_state()

    # we need the next action as well since Q(s,a) depends on Q(s',a')
    # if s2 not in policy then it's a terminal state, all Q are 0
    # the difference between SARSA and Q-Learning is with Q-Learning
    # we will use this max[a']{ Q(s',a')} in our update
    # even if we do not end up taking this action in the next step
    a1, max_Q1 = max_dict(Q[s1])
    # Q[s, a] = Q[s, a] + alpha*(r + gamma*max[a']{Q[s', a']} - Q[s, a])
    # here we use alpha as adaptive learning rate like AdaGrad and
    # RMSprop in DNN
    # in this way when epsilon decreases for each episode it may miss the
    # states which have never occur
    # adaptive alpha will be high for such states and hence keeping the balance
    sa_count[s][action] += 0.005
    Q[s][a] = (Q[s][a] + (alpha/sa_count[s][action]) * (r + gamma*max_Q1 -
                                                        Q[s][a]))
    # we would like to know how often Q(s) has been updated too
    # update_counts[s] = update_counts.get(s,0) + 1

    # set next state as current state
    s = s1
    # update next action as current action
    a = a1

    return Q, s, a, warehouse_log, sa_count


def train_RL_agent():
    # train an RL agent and return Policy (Q value metrics)
    # and the training dataframe

    # supplychain object
    warehouse = Warehouse()

    # set initial Q values for all the states
    states = warehouse.all_states()
    Q = np.zeros((len(states), len(all_possible_actions)))
    update_counts = {}
    sa_count = {}
    for s in states:
        sa_count[s] = {}
        for a in all_possible_actions:
            sa_count[s][a] = 1  # set initial count to be 0

    # dataframe to store data while training
    warehouse_log = pd.DataFrame({
                    'Current_Inventory': [],
                    'Backlog_Orders': [],
                    'Incoming_Delivery': [],
                    'Outgoing_Delivery': [],
                    'Outgoing_Order': [],
                    'Incoming_Order': [],
                    'Closing_Inventory': [],
                    'Closing_Backlog': [],
                    'State': [],
                    'Pending_Delivery': [],
                    'Total_Cost': [],
                    'week': []
                })

    # get incoming order data
    mean, sd = get_incoming_order_stats(incoming_order_csv, io_column)

    n = 1
    # repeat for n episodes
    for episode in tqdm(range(episodes)):
        # decaying epsilon for explore exploit of choosing action
        if episode % 200 == 0:
            eps = 1/n
            n += 1

        # initialize warehouse
        warehouse = Warehouse()

        # choose an action based on max Q value of current state
        a = max_dict(Q[s])[0]

        # get incoming order
        warehouse.incoming_order = get_incoming_order(mean, sd)

        # update outgoing delivery and next inventory, backlog
        warehouse.update()

        # get current state of warehouse
        s = warehouse.current_state()

        # loop until one episode is over
        while not warehouse.year_over():
            # run Q learning and get the new dataframe, state, action
            # and Q table
            Q, s, a, warehouse_log, sa_count = Q_learning(
                                            s, a, Q, warehouse,
                                            eps, warehouse_log, mean, sd,
                                            sa_count)

    # determine the policy from Q*
    # initialize policy, V
    policy, V = {}, {}
    for s in range(-n_bins, n_bins+1):
        policy[s] = all_possible_actions[max_dict(Q[s])[0]]

    print('Action Space Size:', len(all_possible_actions)*len(states))

    return policy, Q, warehouse_log


if __name__ == '__main__':

    # read configuration file
    config = configparser.ConfigParser()
    config.read('config.conf')

    # discount factor
    gamma = eval(config['model_params']['gamma'])

    # incoming order parameters
    # incoming order csv
    incoming_order_csv = eval(config['supply_chain_params']['csv_file_path'])

    # incoming order column
    io_column = eval(config['supply_chain_params']['io_column'])

    # all possible actions at any state of the warehouse
    all_possible_actions = eval(config['supply_chain_params']
                                ['all_possible_actions'])

    # number of levels to divide the total states into
    # -n_bins (for backlog) to +nbins (for inventory)
    n_bins = eval(config['supply_chain_params']['n_bins'])

    # learning rate
    alpha = eval(config['model_params']['alpha'])

    # minimum cost of inventory
    min_cost = eval(config['supply_chain_params']['min_cost'])

    # maximum inventory that I can hold in the store
    max_inventory = eval(config['supply_chain_params']['max_inventory'])

    # define delivery delay from upstream in weeks
    time_delay = eval(config['supply_chain_params']['time_delay'])

    # get random incoming deliveries or not
    random_incoming_delivery = eval(config['supply_chain_params']
                                    ['random_incoming_delivery'])

    # total number of weeks to consider an episode is over
    number_of_weeks = eval(config['supply_chain_params']['number_of_weeks'])

    # total number of episode
    episodes = eval(config['supply_chain_params']['episodes'])

    print('Initial incoming delivery:', 10)
    print('Max Inventory/Backlog Possible:', max_inventory)
    print('Outgoing Orders Possible:', all_possible_actions)
    print("Transport delay in week:", time_delay)

    # start training of RL agent
    policy, Q, warehouse_log = train_RL_agent()

    # save pickle file of policy
    with open('policy.pickle', 'wb') as file:
        pickle.dump(policy, file)

    # save pickle file of Q value
    with open('Q_value.pickle', 'wb') as file:
        pickle.dump(Q, file)

    # number of levels to divide the total states into
    # -n_bins (for backlog) to +nbins (for inventory)
    n_bins = eval(config['supply_chain_params']['n_bins'])

    # learning rate
    alpha = eval(config['model_params']['alpha'])

    # minimum cost of inventory
    min_cost = eval(config['supply_chain_params']['min_cost'])

    # maximum inventory that I can hold in the store
    max_inventory = eval(config['supply_chain_params']['max_inventory'])

    # define delivery delay from upstream in weeks
    time_delay = eval(config['supply_chain_params']['time_delay'])

    # get random incoming deliveries or not
    random_incoming_delivery = eval(config['supply_chain_params']
                                    ['random_incoming_delivery'])

    # total number of weeks to consider an episode is over
    number_of_weeks = eval(config['supply_chain_params']['number_of_weeks'])

    # total number of episode
    episodes = eval(config['supply_chain_params']['episodes'])

    print('Initial incoming delivery:', 10)
    print('Max Inventory/Backlog Possible:', max_inventory)
    print('Outgoing Orders Possible:', all_possible_actions)
    print("Transport delay in week:", time_delay)

    # start training of RL agent
    policy, Q, warehouse_log = train_RL_agent()

    # save pickle file of policy
    with open('policy.pickle', 'wb') as file:
        pickle.dump(policy, file)

    # save pickle file of Q value
    with open('Q_value.pickle', 'wb') as file:
        pickle.dump(Q, file)
