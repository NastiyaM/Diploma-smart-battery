import os
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import *
from battery_controller import BatteryContoller
from battery import Battery
from simulate import Simulation
import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#from matplotlib import rc

import matplotlib.gridspec as gridspec
#from matplotlib import rc


class PvBatteryEnv(object):
    nb_time_steps = 959

    battery_charge = None
    money_saved = None
    current_timestep_idx = None

    actual_previous_load = None
    actual_previous_pv = None

    simulation = None
    battery_controller = None

    site_id = None
    battery_id = None
    period_id = None

    def __init__(self, site_id=1, period_id=1, battery_id=1):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed
                                  to be 0.
        """
        self.actual_previous_pv = []
        self.actual_previous_load = []
        self.battery_charge = []
        self.battery_energy = []
        self.grid_energy = []
        self.money_saved = []
        self.score = []
        self.current_timestep_idx = 0
        simulation_dir = (Path(__file__) / os.pardir / os.pardir).resolve()
        data_dir = "/Users/anastasiamarkelova/PycharmProjects/Smart energy/data"

        # load available metadata to determine the runs
        metadata_path = data_dir +'/metadata.csv'
        metadata = pd.read_csv(metadata_path, index_col=0,sep = ';')

        self.site_id = site_id
        self.period_id = period_id
        self.battery_id = battery_id

        site_data_path = data_dir +"/submit/"+f"{self.site_id}.csv"
        site_data = pd.read_csv(site_data_path, parse_dates=['timestamp'],
                                index_col='timestamp',sep = ';')

        parameters = metadata.loc[self.site_id]
        battery = Battery(capacity=parameters[f"Battery_{self.battery_id}_Capacity"] * 1000,
                          charging_power_limit=parameters[f"Battery_{self.battery_id}_Power"] * 1000,
                          discharging_power_limit=-parameters[f"Battery_{self.battery_id}_Power"] * 1000,
                          charging_efficiency=parameters[f"Battery_{self.battery_id}_Charge_Efficiency"],
                          discharging_efficiency=parameters[
                              f"Battery_{self.battery_id}_Discharge_Efficiency"])

        # n_periods = site_data.period_id.nunique() # keep as comment for now, might be useful later
        g_df = site_data[site_data.period_id == self.period_id]

        self.simulation = Simulation(g_df, battery, self.site_id)
        self.battery_controller = BatteryContoller()

    def step(self):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Returns
            observation (object): Agent's observation of the current environment.
            done (boolean): Whether the episode has ended, in which case further step() calls
                            will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging,
                         and sometimes learning).
        """

        money_saved, grid_energy = self.simulation.simulate_timestep(
            self.battery_controller,
            self.simulation.data.index[self.current_timestep_idx],
            self.simulation.data.iloc[self.current_timestep_idx])

        self.grid_energy.append(grid_energy)
        self.battery_charge.append(self.simulation.battery.current_charge)
        self.battery_energy.append(self.simulation.battery.capacity *
                                   self.simulation.battery.current_charge)
        self.money_saved.append(money_saved)
        self.score.append(self.simulation.score)
        self.actual_previous_load.append(self.simulation.actual_previous_load)
        self.actual_previous_pv.append(self.simulation.actual_previous_pv)

        self.current_timestep_idx += 1

    def render(self, mode='human', close=False):
        plt.subplot(2, 3, 1)
        plt.plot(self.battery_charge)
        plt.legend(['Charge', 'Target'])
        plt.title('Battery')

        plt.subplot(2, 3, 2)
        plt.plot(np.array(self.actual_previous_load) - np.array(self.actual_previous_pv))
        plt.title('Load - PV')

        plt.subplot(2, 3, 3)
        plt.plot(self.money_saved)
        plt.plot(np.cumsum(self.money_saved))
        plt.legend(['Instantaneous', 'Cumulative'])
        plt.title('Agregate reward')

        ax = plt.subplot(2, 3, 4)
        self.simulation.data[['price_sell_00', 'price_buy_00']].plot(ax=ax)

        plt.subplot(2, 3, 5)
        plt.plot(self.score)
        plt.title('Simulation score')

    def save(self):
        t = (self.battery_charge,
             self.battery_energy,
             self.simulation.data.actual_consumption,
             self.simulation.data.actual_pv,
             self.grid_energy,
             self.simulation.data['price_sell_00'],
             self.simulation.data['price_buy_00'],
             self.money_saved,
             self.score)

        # save data in pickle file
        simulation_dir = (Path(__file__) / os.pardir / os.pardir).resolve()
        pickle_path = "/Users/anastasiamarkelova/PycharmProjects/Smart energy/analysis_of_results/env_DDPG/"+f"env_s{self.site_id}_b{self.battery_id}_p{self.period_id}.p"
        with open(pickle_path, 'wb') as f:
            pickle.dump(t, f)


if __name__ == '__main__':
    simulation_dir = (Path(__file__) / os.pardir / os.pardir).resolve()
    for site_id in [1]:
        for period_id in [1]:
            for battery_id in [1]:
    #site_id = 1
            #period_id = 1
    #battery_id = 1

                env = PvBatteryEnv(site_id=site_id, period_id=period_id, battery_id=battery_id)
                for _ in range(96):
                    env.step()
                    env.save()
                    id_site = site_id
                    bat_id = battery_id
                    id_period = period_id
                    pickle_path = "/Users/anastasiamarkelova/PycharmProjects/Smart energy/analysis_of_results/DDPG/"+f"norm_s{str(id_site)}.p"
                    with open(pickle_path, "rb") as f:
                        norm = pickle.load(f)
                    pickle_path = "/Users/anastasiamarkelova/PycharmProjects/Smart energy/analysis_of_results/env_DDPG/"+f"env_s{id_site}_b{bat_id}_p{id_period}.p"
                    with open(pickle_path, "rb") as f:
                        t = pickle.load(f)
                    (battery_charge, battery_energy, actual_consumption, actual_pv, grid_energy, price_sell, price_buy,
                     money_saved, score) = t
                    # plot params
                    sns.set_style("white", {"axes.facecolor": ".98"})

                    fig = plt.figure(tight_layout=False, figsize=(16.5, 8.5))

                    fontsize_ns = 24
                    fontsize_sub = 20

                    c1 = sns.xkcd_rgb['blue']
                    c2 = sns.xkcd_rgb['green']
                    c3 = sns.xkcd_rgb['aqua']

                    c4 = sns.xkcd_rgb['red']
                    c5 = sns.xkcd_rgb['purple']
                    # c6 = sns.xkcd_rgb['orange']

                    savings = np.cumsum(money_saved) / np.sum(money_saved) / 20.

                    plot(actual_consumption.values / 300, c1)
                    actual_pv = actual_pv.values / 300
                    plot(actual_pv, c2)

                    plot(np.array(battery_energy) / 1000, c3)
                    plt.ylabel('Энергия(kW)', fontsize=18)
                    plt.tick_params(labelsize=14)
                    plt.ylim([0, 500])
                    plt.xlim([0, 96])

                    twinx()

                    plot(price_buy.values * norm['price'], c1)
                    plot(price_buy.values * norm['price'], c2)
                    plot(price_buy.values * norm['price'], c3)
                    plot(price_buy.values * norm['price'], c4)
                    plot(price_sell.values * norm['price'], c5, linestyle='--')
                    # plot(savings, c6)
                    plt.legend(['Нагрузка', 'PV генерация', 'SoC', 'Цена покупки', 'Цена продажи'], fontsize=16, loc=1)
                    plt.ylabel('Цена (€/kW)', fontsize=18)
                    plt.tick_params(labelsize=14)
                    plt.tick_params(labelsize=14)
                    plt.ylim([min(savings), 1.7 * max(savings)])
                    plt.xlim([0, 96])

