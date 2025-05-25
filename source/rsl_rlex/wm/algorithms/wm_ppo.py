from __future__ import annotations
import torch
from rsl_rl.algorithms import PPO
from rsl_rlex.wm.modules import wm_modules
from rsl_rlex.wm.storage import wm_storage
from torch.nn import functional as F
import torch.optim as optim
from torch.nn import utils

class WMPPO(PPO):

    actor_critic: wm_modules.WMActorCritic
    storage: wm_storage.RolloutStorage# type: ignore
    transition : wm_storage.RolloutStorage.Transition

    def __init__(self, actor_critic, alg_cfg, device="cpu"):
        super(WMPPO, self).__init__(
            actor_critic,
            num_learning_epochs=alg_cfg['num_learning_epochs'],
            num_mini_batches=alg_cfg['num_mini_batches'],
            clip_param=alg_cfg['clip_param'],
            gamma=alg_cfg['gamma'],
            lam=alg_cfg['lam'],
            value_loss_coef=alg_cfg['value_loss_coef'],
            entropy_coef=alg_cfg['entropy_coef'],
            learning_rate=alg_cfg['learning_rate'],
            max_grad_norm=alg_cfg['max_grad_norm'],
            use_clipped_value_loss=alg_cfg['use_clipped_value_loss'],
            schedule=alg_cfg['schedule'],
            desired_kl=alg_cfg['desired_kl'],
            device=device,
            normalize_advantage_per_mini_batch=False,
            # RND parameters
            rnd_cfg=None,
            # Symmetry parameters
            symmetry_cfg=None,
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters()
            , lr=alg_cfg['learning_rate'])

    def init_storage(self, num_envs, num_transitions_per_env, action_shape, obs):
        self.storage = wm_storage.RolloutStorage(
            num_envs,
            num_transitions_per_env,
            action_shape,
            obs,
            self.device,
        )
        self.transition = wm_storage.RolloutStorage.Transition()


    def act(self, obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_continuous_loss = 0
        mean_discrete_loss = 0
        mean_reconstruct_loss = 0

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            #generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            raise 'don\'t support recurrent'

        hid_masks = None

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            #######################
            # original batch size
            continuous = self.actor_critic.continuous_observations(obs_batch)
            discrete = self.actor_critic.discrete_observations(obs_batch)
            continuous_batch, discrete_batch = self.actor_critic.reconstruction(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[2])

            #
            reconstruct = torch.cat((continuous_batch.detach(), discrete_batch.detach()), dim = -1)
            valid = reconstruct.shape[1]

            continuous_loss = F.mse_loss(continuous[:, :valid], continuous_batch)
            discrete_loss = F.binary_cross_entropy(discrete[:, :valid], discrete_batch)
            reconstruct_loss = continuous_loss + discrete_loss * 0.3

            original_batch_size = reconstruct.shape[1]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], is_critic=False
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            hid_states = hid_states_batch[0]
            if isinstance(hid_states, (list, tuple)):
                result = []
                for _hid in hid_states:
                    result.append(_hid[:, :valid])
                hid_states = result
            else:
                hid_states = hid_states[:, :valid]

            if hid_masks is None:
                hid_masks = torch.ones(*reconstruct.shape[:2], dtype=torch.bool, device = reconstruct.device)

            self.actor_critic.act(reconstruct, masks=hid_masks, hidden_states=hid_states)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + reconstruct_loss

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], is_critic=False
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], is_critic=False
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            mean_continuous_loss += continuous_loss.item()
            mean_discrete_loss += discrete_loss.item()
            mean_reconstruct_loss += reconstruct_loss.item()

            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        mean_continuous_loss /= num_updates
        mean_discrete_loss /= num_updates
        mean_reconstruct_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss, \
                mean_continuous_loss, mean_discrete_loss, mean_reconstruct_loss
