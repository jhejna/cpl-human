"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import math
import policy
from common import Batch, InfoDict, Model, PRNGKey, Params



ACTION_MIN, ACTION_MAX = -1, 1


def safe_norm(x, **kwargs):
    # l2 norm with gradient set to 0 when norm is 0
    return jnp.linalg.norm(jnp.where(x == 0, 0, x), **kwargs)


def update_actor(key: PRNGKey, actor: Model, batch: Batch, lambd: float,
                      dist_temperature: float) -> Tuple[Model, InfoDict]:
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # calculate distance
        policy_actions = actor.apply({"params": actor_params},
                           batch.observations,
                           training=True,
                           rngs={"dropout": key},
                           )
        actions = batch.actions
        scores = batch.scores

        # normalize action differences
        step_diffs = (policy_actions - actions) / (ACTION_MAX - ACTION_MIN)

        step_distances = safe_norm(step_diffs, axis=2)
        traj_distances = (step_distances * batch.masks).sum(axis=1) / batch.masks.sum(axis=1)
        
        # calculate score
        indices = jnp.argsort(scores, axis=0)
        indices = jnp.flip(indices) # to descending order

        distances_sorted = traj_distances[indices] / dist_temperature
        distances_sum = jnp.exp(-distances_sorted) + jnp.exp(-lambd * distances_sorted).reshape(-1)
        pair_score = jnp.exp(-distances_sorted) / distances_sum

        log_score_triu = jnp.triu(jnp.log(pair_score), k=1)

        mask = log_score_triu != 0
        score = (log_score_triu).sum() / mask.sum()
        actor_loss = - score
        return actor_loss, {'actor_loss': actor_loss, 'score': score}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info

def update_actor_cpl(key: PRNGKey, actor: Model, batch: Batch, lambd: float,
                      dist_temperature: float) -> Tuple[Model, InfoDict]:
    
    def cpl_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # calculate distance
        policy_actions = actor.apply({"params": actor_params},
                           batch.observations,
                           training=True,
                           rngs={"dropout": key},
                           )
        actions = batch.actions  # (B, S, D)
        scores = jnp.squeeze(batch.scores, axis=-1)  # (B,)

        # Compute the advantage
        lp = (-((policy_actions - actions) / (ACTION_MAX - ACTION_MIN))**2).sum(axis=2) # (B, S)
        segment_lp = (lp * batch.masks).sum(axis=1) / batch.masks.sum(axis=1) # Note that this is average

        # To choose the dist temp parameter  t such thast alpha * adv = (adv / 100) / t
        # => alpha = 1/100t => t = 1 / (100*alpha)
        # By default CPL divides the distances by 2 (act range), then by 100 (timesteps), then by 0.1 (temp)
        # this means the default alpha is 0.05, and we would set dist_temp 0.2
        adv = segment_lp / dist_temperature

        idx = jnp.argsort(scores, axis=0)
        adv_sorted = adv[idx]

        logits = jnp.expand_dims(adv_sorted, axis=0) - lambd * jnp.expand_dims(adv_sorted, axis=1)
        max_val = jnp.clip(-logits, a_min=0, a_max=None)
        loss = jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val)) + max_val

        loss = jnp.triu(loss, k=1)
        mask = loss != 0.0
        loss = loss.sum() / mask.sum()

        return loss, {'actor_loss': loss, 'score': -loss}

    new_actor, info = actor.apply_gradient(cpl_loss_fn)
    return new_actor, info

def update_actor_cpl_prob(key: PRNGKey, actor: Model, batch: Batch, lambd: float,
                      dist_temperature: float) -> Tuple[Model, InfoDict]:
    
    def cpl_loss_fn_prob(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # calculate distance
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        lp = dist.log_prob(batch.actions).sum(axis=2) # (B, S)
        scores = jnp.squeeze(batch.scores, axis=-1)  # (B,)

        # Compute the advantage
        segment_lp = (lp * batch.masks).sum(axis=1) / batch.masks.sum(axis=1) # Note that this is average

        # To choose the dist temp parameter  t such thast alpha * adv = (adv / 100) / t
        # => alpha = 1/100t => t = 1 / (100*alpha)
        # By default CPL divides the distances by 2 (act range), then by 100 (timesteps), then by 0.1 (temp)
        # this means the default alpha is 0.05, and we would set dist_temp 0.2
        adv = segment_lp / dist_temperature

        idx = jnp.argsort(scores, axis=0)
        adv_sorted = adv[idx]

        logits = jnp.expand_dims(adv_sorted, axis=0) - lambd * jnp.expand_dims(adv_sorted, axis=1)
        max_val = jnp.clip(-logits, a_min=0, a_max=None)
        loss = jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val)) + max_val

        loss = jnp.triu(loss, k=1)
        mask = loss != 0.0
        loss = loss.sum() / mask.sum()

        return loss, {'actor_loss': loss, 'score': -loss}

    new_actor, info = actor.apply_gradient(cpl_loss_fn_prob)
    return new_actor, info


def update_actor_bc(key: PRNGKey, actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    
    def bc_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        policy_actions = actor.apply({"params": actor_params},
                           batch.observations,
                           training=True,
                           rngs={"dropout": key},
                           )
        actions = batch.actions  # (B, S, D)
        scores = jnp.squeeze(batch.scores, axis=-1)  # (B,)

        # Compute the advantage
        mse = ((policy_actions - actions)**2).sum(axis=2) # (B, S) 
        loss = (mse * batch.masks).sum() / batch.masks.sum()

        return loss, {'actor_loss': loss, 'score': -loss}

    new_actor, info = actor.apply_gradient(bc_loss_fn)
    return new_actor, info

def update_actor_bc_prob(key: PRNGKey, actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    
    def bc_loss_fn_prob(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        lp = dist.log_prob(batch.actions).sum(axis=2) # (B, S)

        loss = -lp
        loss = (mse * batch.masks).sum() / batch.masks.sum()

        return loss, {'actor_loss': loss, 'score': -loss}

    new_actor, info = actor.apply_gradient(bc_loss_fn_prob)
    return new_actor, info

@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, batch: Batch, lambd: float, dist_temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, batch, lambd, dist_temperature)

    return rng, new_actor, {
        **actor_info
    }

@jax.jit
def _update_jit_cpl(
    rng: PRNGKey, actor: Model, batch: Batch, lambd: float, dist_temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_cpl(key, actor, batch, lambd, dist_temperature)

    return rng, new_actor, {
        **actor_info
    }

@jax.jit
def _update_jit_cpl_prob(
    rng: PRNGKey, actor: Model, batch: Batch, lambd: float, dist_temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_cpl_prob(key, actor, batch, lambd, dist_temperature)

    return rng, new_actor, {
        **actor_info
    }

@jax.jit
def _update_jit_bc_prob(
    rng: PRNGKey, actor: Model, batch: Batch
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_bc_prob(key, actor, batch)

    return rng, new_actor, {
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "",
                 lambd: float = 1.0,
                 dist_temperature: float = 1.0,
                 cpl=False,
                 bc_steps: int = 0,
                 probabilistic: bool = False,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.lambd = lambd
        self.dist_temperature = dist_temperature
        self.cpl = cpl
        self.bc_steps = bc_steps
        self.probabilistic = probabilistic

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        action_dim = actions.shape[-1]
        if self.probabilistic:
            actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        else:
            actor_def = policy.DeterministicPolicy(hidden_dims,
                                                action_dim,
                                                dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimizer = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimizer = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimizer)

        self.actor = actor
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray, temperature: float = 1.0,
                       **kwargs,
                       ) -> jnp.ndarray:

        if self.probabilistic:
            actions = policy.sample_actions(self.actor.apply_fn,
                                                self.actor.params, observations, temperature)
        else:
            actions = policy.sample_actions_det(self.actor.apply_fn,
                                                self.actor.params, observations)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, step: int) -> InfoDict:
        if self.cpl:
            if step < self.bc_steps:
                if self.probabilistic:
                    new_rng, new_actor, info = _update_jit_bc_prob(
                        self.rng, self.actor, batch)
                else:
                    new_rng, new_actor, info = _update_jit_bc(
                        self.rng, self.actor, batch)
            else:
                if self.probabilistic:
                    new_rng, new_actor, info = _update_jit_cpl_prob(
                        self.rng, self.actor, batch, self.lambd, self.dist_temperature)
                else:
                    new_rng, new_actor, info = _update_jit_cpl(
                        self.rng, self.actor, batch, self.lambd, self.dist_temperature)
        else:
            new_rng, new_actor, info = _update_jit(
                self.rng, self.actor, batch, self.lambd, self.dist_temperature)

        self.rng = new_rng
        self.actor = new_actor

        return info
