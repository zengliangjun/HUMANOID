from __future__ import annotations

import torch
from rsl_rl.utils import split_and_pad_trajectories
from rsl_rl.storage import rollout_storage

class RolloutStorage(rollout_storage.RolloutStorage):
    class Transition:
        def __init__(self):
            self.observations = None

            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def _init_obs(self, obs):
        observations = {}
        for name, value in obs.items():
            observations[name] = torch.zeros(
                self.num_transitions_per_env,
                *value.shape, device=self.device)

        self.observations = observations

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actions_shape,
        obs,
        device="cpu",
        rnd_state_shape=None,
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        self._init_obs(obs)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_encoder = None
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        for name, value in transition.observations.items():
            self.observations[name][self.step].copy_(value)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For PPO
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1


    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return

        # initialize if needed
        if self.saved_hidden_states_a is None:

            # make a tuple out of GRU hidden state sto match the LSTM format
            hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
            hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

            self.saved_hidden_states_a = [
                torch.zeros(self.num_transitions_per_env, *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.num_transitions_per_env, *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        super()._save_hidden_states(hidden_states)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = {}
        for name, value in self.observations.items():
            observations[name] = value.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = {}
                for name, value in observations.items():
                    obs_batch[name] = value[batch_idx]

                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # Yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories = {}
        for name, value in self.observations.items():
            trajectories, trajectory_masks = split_and_pad_trajectories(value, self.dones)
            padded_obs_trajectories[name] = trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]

                obs_batch = {}
                for name, trajectories in padded_obs_trajectories.items():
                    obs_batch[name] = trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj
