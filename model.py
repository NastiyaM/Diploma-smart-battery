import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
from collections import deque
import random
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

class Battery(object):
    """ Used to store information about the battery.
       :param current_charge: is the initial state of charge of the battery
       :param capacity: is the battery capacity in Wh
       :param charging_power_limit: the limit of the power that can charge the battery in W
       :param discharging_power_limit: the limit of the power that can discharge the battery in W
       :param battery_charging_efficiency: The efficiecny of the battery when charging
       :param battery_discharing_efficiecny: The discharging efficiency
    """

    def __init__(self,
                 current_charge=0.0,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.current_charge = current_charge
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency


class Simulation(object):
    """ Handles running a simulation.
    """

    def __init__(self,
                 data,
                 battery,
                 site_id, act):
        """ Creates initial simulation state based on data passed in.
            :param data: contains all the time series needed over the considered period
            :param battery: is a battery instantiated with 0 charge and the relevant properties
            :param site_id: the id for the site (building)
        """
        self.act = act
        self.data = data

        # building initialization

        # initialize money at 0.0
        self.money_spent = 0.0
        self.money_spent_without_battery = 0.0

        # battery initialization
        self.battery = battery
        self.reward = 0.0

    def run(self):
        """ Executes the simulation by iterating through each of the data points
            It returns both the electricity cost spent using the battery and the
            cost that would have been incurred with no battery.
        """
        timestep = self.data
        if pd.notnull(timestep.actual_consumption):
            self.simulate_timestep(timestep, self.act)
        # return self.money_spent, self.money_spent_without_battery
        return self.battery.current_charge, self.reward, (self.money_spent - self.money_spent_without_battery) / np.abs(
            self.money_spent_without_battery)

    def simulate_timestep(self, timestep, act):
        """ Executes a single timestep using `battery_controller` to get
            a proposed state of charge and then calculating the cost of
            making those changes.
            :param battery_controller: The battery controller
            :param current_time: the timestamp of the current time step
            :param timestep: the data available at this timestep
        """
        # get proposed state of charge from the battery controller
        charging_efficiency = self.battery.charging_efficiency
        discharging_efficiency = 1. / self.battery.discharging_efficiency
        capacity = self.battery.capacity
        if self.act > 0:
            proposed_state_of_charge = self.battery.current_charge + self.act * charging_efficiency
        else:
            proposed_state_of_charge = self.battery.current_charge + self.act * discharging_efficiency

        proposed_state_of_charge = self.battery.current_charge + self.act

        # get energy required to achieve the proposed state of charge
        grid_energy, battery_energy_change = self.simulate_battery_charge(self.battery.current_charge,
                                                                          proposed_state_of_charge,
                                                                          timestep.actual_consumption / 1000,
                                                                          timestep.actual_pv / 1000, timestep)

        grid_energy_without_battery = timestep.actual_consumption - timestep.actual_pv

        # buy or sell energy depending on needs
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00
        price_without_battery = timestep.price_buy_00 if grid_energy_without_battery >= 0 else timestep.price_sell_00

        # calculate spending based on price per kWh and energy per Wh
        self.money_spent += grid_energy * (price / 1000.)
        self.money_spent_without_battery += grid_energy_without_battery * (price_without_battery / 1000.)

        # update current state of charge
        self.battery.current_charge += battery_energy_change
        self.actual_previous_load = timestep.actual_consumption
        self.actual_previous_pv = timestep.actual_pv
        self.reward = self.money_spent
        # return self.battery.current_charge, price

    def simulate_battery_charge(self, initial_state_of_charge, proposed_state_of_charge, actual_consumption, actual_pv,
                                timestep):
        """ Charges or discharges the battery based on what is desired and
            available energy from grid and pv.
            :param initial_state_of_charge: the current state of the battery
            :param proposed_state_of_charge: the proposed state for the battery
            :param actual_consumption: the actual energy consumed by the building
            :param actual_pv: the actual pv energy produced and available to the building
        """
        # charge is bounded by what is feasible
        proposed_state_of_charge = np.clip(proposed_state_of_charge, 0.0, self.battery.capacity)

        # calculate proposed energy change in the battery
        target_energy_change = (proposed_state_of_charge - initial_state_of_charge)

        # efficiency can be different whether we intend to charge or discharge
        if target_energy_change >= 0:
            efficiency = self.battery.charging_efficiency
            target_charging_power = target_energy_change / (efficiency)
        else:
            efficiency = self.battery.discharging_efficiency
            target_charging_power = target_energy_change * efficiency

            # actual power is bounded by the properties of the battery
        actual_charging_power = np.clip(target_charging_power,
                                        self.battery.discharging_power_limit,
                                        self.battery.charging_power_limit)

        # actual energy change is based on the actual power possible and the efficiency
        if actual_charging_power >= 0:
            actual_energy_change = actual_charging_power * efficiency
        else:
            actual_energy_change = actual_charging_power / efficiency

        # what we need from the grid = (the power put into the battery + the consumption) - what is available from pv
        grid_energy = (actual_charging_power + actual_consumption) - actual_pv
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00

        # if positive, we are buying from the grid; if negative, we are selling

        return grid_energy, actual_energy_change

"""
solving using dqn model
"""

class Agent():
    def __init__(self):
        self.memory = deque(maxlen=10000)

        self.gamma = 0.85
        self.epsilon = 5.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = 391
        model.add(Dense(391, input_dim=state_shape, activation="relu"))
        model.add(Dense(300, activation="relu"))
        model.add(Dense(300, activation="relu"))
        model.add(Dense(12))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randrange(12)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 3000
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

"""
solving using actor-critic model
"""

class ActorCritic:
    def __init__(self, sess):
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 0.6
        # self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125
        self.action_space = [i for i in range(-75, 75)]

        self.memory = deque(maxlen=10000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, 1])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=[391])
        h1 = Dense(300, activation='relu')(state_input)
        h2 = Dense(300, activation='relu')(h1)
        output = Dense(1, activation='relu')(h2)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=[391])
        state_h1 = Dense(300, activation='relu')(state_input)
        state_h2 = Dense(300)(state_h1)

        action_input = Input(shape=[1])
        action_h1 = Dense(300)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(300, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], [reward], verbose=0)

    def train(self):
        batch_size = 3000
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        # self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.actor_model.predict(cur_state)
    def save_model(self, fn):
        self.actor_model.save(fn)


if __name__ == '__main__':
    # load available metadata to determine the runs
    metadata_path = 'data/metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0, sep=";")
    metadata = metadata[metadata.index == 1]
    score = {'agent_DQN':[],'agent_DDPG':[]}
    sess = tf.Session()
    K.set_session(sess)
    agent_DQN = Agent()
    agent_DDPG = ActorCritic(sess)
    for agent in (['agent_DQN', 'agent_DDPG']):
        for site_id, parameters in metadata.iterrows():
        # # execute two runs with each battery for every row in the metadata file:
            site_data_path = "data/train.csv"
            site_data = pd.read_csv(site_data_path, parse_dates=['timestamp'], index_col='timestamp', sep=";")
            for batt_id in [1, 2]:
            # create the battery for this run
            # (Note: Quantities in kW are converted to watts here)
                batt = Battery(capacity=parameters[f"Battery_{batt_id}_Capacity"],
                            charging_power_limit=parameters[f"Battery_{batt_id}_Power"],
                            discharging_power_limit=-parameters[f"Battery_{batt_id}_Power"],
                            charging_efficiency=parameters[f"Battery_{batt_id}_Charge_Efficiency"],
                            discharging_efficiency=parameters[f"Battery_{batt_id}_Discharge_Efficiency"])
                d = {1: batt.discharging_power_limit, 2: batt.discharging_power_limit * 0.5,
                    3: batt.discharging_power_limit * 0.75, 4: batt.discharging_power_limit * 0.25,
                    5: batt.charging_power_limit, 6: batt.charging_power_limit * 0.4, 7: batt.charging_power_limit * 0.25,
                    8: batt.charging_power_limit * 0.5, 9: batt.charging_power_limit * 0.75,
                    10: batt.charging_power_limit * 0.1, 11: batt.discharging_power_limit * 0.1, 0: 0}
            # execute the simulation for each simulation period in the data
                episodes = 10000
                state_size = 391
                action_size = 12
                steps = []
                i = 0
                for index_episode in range(episodes):
                    batt.current_charge = 0
                    state = list(site_data.iloc[i])
                    state.append(batt.current_charge)
                    state = np.reshape(state, [1, state_size])
                    done = False
                    while not done:
                        if i == (len(site_data) - 1):
                            break
                        Timestamp = site_data.index[i]
                        action = agent.act(state)
                        if agent == 'agent_DQN':
                            pw = d[action]
                        sim = Simulation(site_data.iloc[i + 1], batt, site_id, pw)
                        batt.current_charge, rew, sc = sim.run()
                        next_state = list(site_data.iloc[i + 1])
                        next_state.append(batt.current_charge)
                        reward = rew
                        if Timestamp.hour == 00 and Timestamp.minute == 00:
                            done = True
                        else:
                            done = False
                        next_state = np.reshape(next_state, [1, state_size])
                        agent.remember(state, action, reward, next_state, done)
                        i += 1
                        if agent == 'agent_DQN':
                            agent.replay()
                            agent.target_train()
                        else:
                            agent.train()
                        state = next_state
                    sc = 0
                    for j in range(96):
                        state_test = list(site_data.iloc[j])
                        state_test.append(0)
                        state_test = np.reshape(state, [1, state_size])
                        action_test = agent.act(state)
                        pw = d[action]
                        sim = Simulation(site_data.iloc[j + 1], batt, site_id, pw)
                        batt.current_charge, rew, sc = sim.run()
                        sc += rew
                    score[agent].append(sc/j)
        if agent == 'agent_DQN':
            agent.save_model("analysis_of_results/DQN/{}.h5".format(site_id))
        else:
            agent.save_model("analysis_of_results/DDPG/{}.h5".format(site_id))
    x = [i for i in range(10000)]
    plt.figure(figsize=(12, 7))
    plt.plot(x, score['agent_DQN'])
    plt.plot(x, score['agent_DDPG'])
    plt.legend(['DQN','DDPG'], fontsize=16, loc=4)
    plt.ylabel('Средняя награда', fontsize=18)
    plt.xlabel('Эпизод', fontsize=18)
    plt.savefig('episode_10000', format='png')
    x = [i for i in range(100)]
    plt.figure(figsize=(12, 7))
    plt.plot(x, score['agent_DQN'][0:100])
    plt.plot(x, score['agent_DDPG'][0:100])
    plt.legend(['DQN','DDPG'], fontsize=16, loc=4)
    plt.ylabel('Средняя награда', fontsize=18)
    plt.xlabel('Эпизод', fontsize=18)
    plt.savefig('episode_100', format='png')
    x = [i for i in range(100)]
    plt.figure(figsize=(12, 7))
    plt.plot(x, score['agent_DQN'][9890:9990])
    plt.plot(x, score['agent_DDPG'][9890:9990])
    plt.legend(['DQN','DDPG'], fontsize=16, loc=4)
    plt.ylabel('Средняя награда', fontsize=18)
    plt.xlabel('Эпизод', fontsize=18)
    plt.savefig('episode_9900', format='png')
