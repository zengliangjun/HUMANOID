from collections import defaultdict
from multiprocessing import Process
import torch


from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

class LoggerBase:
    def __init__(self, env):
        self.env: ManagerBasedRLEnv = env
        self.asset: Articulation = env.scene["robot"]
        self.dt = self.env.step_dt
        self.robot_id = 0

        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.num_episodes = 0
        self.plot_process = None

    def log_step(self, actions, extra):
        states = self._calcute_states(actions, extra)
        self._log_states(states)

        num_episodes = torch.sum(self.env.reset_buf).item()
        if num_episodes>0:

            if "episode" in extra:
                ep_infos = extra["episode"]
            elif "log" in extra:
                ep_infos = extra["log"]

            #self._log_rewards(ep_infos, num_episodes)
            #self._print_rewards()
            self._plot_states()
            self._reset()

    def _log_state(self, key, value):
        self.state_log[key].append(value)

    def _log_states(self, dict):
        for key, value in dict.items():
            self._log_state(key, value)

    def _plot_states(self):
        self.plot_process = Process(target=self._plot, args=([self.state_log]))
        self.plot_process.start()

    def _reset(self):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.num_episodes = 0

    def _calcute_states(self, actions, extra):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def _plot(self, state):
        raise NotImplementedError("This method should be implemented in a subclass.")
