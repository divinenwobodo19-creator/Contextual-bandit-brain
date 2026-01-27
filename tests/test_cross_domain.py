import numpy as np

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def run_with_custom_contexts(num_steps, context_sampler, env, brain):
    rewards = []
    for _ in range(num_steps):
        x = context_sampler(env._rng, env.d)
        brain.observe(x)
        a = brain.choose_action()
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)
    return np.array(rewards)


def sampler_uniform(rng, d):
    return rng.uniform(low=-1.0, high=1.0, size=d).astype(float)


def sampler_normal(rng, d):
    return rng.normal(loc=0.0, scale=1.0, size=d).astype(float)


def sampler_laplace(rng, d):
    return rng.laplace(loc=0.0, scale=1.0, size=d).astype(float)


def test_domain_agnostic_behavior():
    d, num_actions, alpha = 10, 6, 0.4
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.2, seed=123)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1800

    rewards_uniform = run_with_custom_contexts(steps, sampler_uniform, env, brain)
    avg_u_initial = float(np.mean(rewards_uniform[:300]))
    avg_u_final = float(np.mean(rewards_uniform[-300:]))
    assert avg_u_final > avg_u_initial

    rewards_normal = run_with_custom_contexts(steps, sampler_normal, env, brain)
    avg_n_initial = float(np.mean(rewards_normal[:300]))
    avg_n_final = float(np.mean(rewards_normal[-300:]))
    assert avg_n_final > avg_n_initial

    rewards_laplace = run_with_custom_contexts(steps, sampler_laplace, env, brain)
    avg_l_initial = float(np.mean(rewards_laplace[:300]))
    avg_l_final = float(np.mean(rewards_laplace[-300:]))
    assert avg_l_final > avg_l_initial

    # Consistency: the brain relies only on numeric context and rewards
    assert isinstance(avg_u_final, float) and isinstance(avg_n_final, float) and isinstance(avg_l_final, float)

